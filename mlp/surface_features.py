"""
Computes and exports feature vectors for three hyperbolic surfaces:
  - X0N_features : feature matrix for X_0(N) over squarefree N
  - bolza_features: feature vector for the Bolza surface
  - klein_features: feature vector for the Klein quartic
 
Each feature vector has the form [genus, volume, L_1, L_2, ...].
"""
 
import math
import numpy as np
import torch

squarefree_Ns = [n for n in range(1, 106) if all(n % (p*p) != 0 for p in range(2, int(n**0.5) + 1))]
# =============================================================================
#  1.  Primitive geodesic lengths for X_0(N)
# =============================================================================
 
def simplify_sqrt(n):
    """
    Simplifies sqrt(n) by pulling out perfect square factors.
    Returns a tuple (coefficient, radicand) such that sqrt(n) = coefficient * sqrt(radicand).
    """
    if n < 0:
        raise ValueError("Cannot simplify sqrt of negative number in this context.")
    if n == 0:
        return (0, 0)
 
    coefficient = 1
    radicand = n
    i = 2
    while i * i <= radicand:
        if radicand % (i * i) == 0:
            coefficient *= i
            radicand //= (i * i)
        else:
            i += 1
    return (coefficient, radicand)
 
 
def get_representative_matrix_entries(trace):
    """
    Returns a simple representative primitive hyperbolic matrix for a given trace
    as a tuple (a,b,c,d), with a+d = trace and ad-bc = 1.
    """
    if trace % 2 == 0:
        k = trace // 2
        a, b, c, d = k, 1, k*k - 1, k
    else:
        k = (trace - 1) // 2
        a, b, c, d = k + 1, 1, k*(k + 1) - 1, k
    return (a, b, c, d)
 
 
def format_matrix(entries):
    """Formats a matrix tuple (a,b,c,d) as a string."""
    a, b, c, d = entries
    return f"[[{a}, {b}], [{c}, {d}]]"
 
 
def generate_primitive_geodesics(num_geodesics):
    """
    Generates the lengths of the shortest closed geodesics for SL(2,Z),
    including trace, a closed-form string representation, and a representative matrix.
    """
    geodesics = []
    non_primitive_traces = set()
 
    current_trace = 3
    max_trace_to_check = 100000  # Heuristic limit
 
    while len(geodesics) < num_geodesics and current_trace <= max_trace_to_check:
        if current_trace in non_primitive_traces:
            current_trace += 1
            continue
 
        discriminant = current_trace**2 - 4
        if discriminant < 0:
            current_trace += 1
            continue
 
        larger_eigenvalue = (current_trace + math.sqrt(discriminant)) / 2.0
        length = 2 * math.log(larger_eigenvalue)
 
        if discriminant == 0:
            closed_form_string = f"2 * log({current_trace / 2})"
        else:
            coeff, rad = simplify_sqrt(discriminant)
            sqrt_part_str = (
                "0" if rad == 0 else
                str(coeff) if rad == 1 else
                f"sqrt({rad})" if coeff == 1 else
                f"{coeff} * sqrt({rad})"
            )
            closed_form_string = f"2 * log(({current_trace} + {sqrt_part_str}) / 2)"
 
        matrix_entries = get_representative_matrix_entries(current_trace)
        matrix_string = format_matrix(matrix_entries)
 
        geodesics.append((length, current_trace, closed_form_string, matrix_string, matrix_entries))
 
        # Filter powers via Chebyshev recurrence
        t_0, t_1 = 2, current_trace
        while True:
            t_next = current_trace * t_1 - t_0
            if t_next > max_trace_to_check:
                break
            non_primitive_traces.add(t_next)
            t_0, t_1 = t_1, t_next
 
        current_trace += 1
 
    geodesics.sort(key=lambda x: x[0])
    return geodesics[:num_geodesics]
 
 
def get_X0N_geodesics(K, max_N):
    """
    Returns an array where index N contains the first K geodesic lengths for X_0(N).
 
    We filter SL(2,Z) geodesics by checking whether the representative matrix lies in Γ_0(N),
    i.e. whether c ≡ 0 (mod N).
    """
    n_needed = 50000
    sl2z_geodesics = generate_primitive_geodesics(n_needed)
 
    lengths = np.array([g[0] for g in sl2z_geodesics])
    c_entries = np.array([g[4][2] for g in sl2z_geodesics])  # lower-left entry c
 
    result = []
 
    for N in range(max_N + 1):
        if N == 0:
            result.append(np.array([]))
            continue
 
        mask = (c_entries % N) == 0
        valid_lengths = lengths[mask]
        result.append(valid_lengths[:K])
 
    return np.array(result, dtype=object)
 
 
# =============================================================================
#  2.  Genus and volume of X_0(N)
# =============================================================================
 
def factorint(N):
    f, n, p = {}, N, 2
    while p * p <= n:
        while n % p == 0:
            f[p] = f.get(p, 0) + 1
            n //= p
        p = 3 if p == 2 else p + 2
    if n > 1:
        f[n] = f.get(n, 0) + 1
    return f
 
 
def phi(N):
    r = 1
    for p, e in factorint(N).items():
        r *= (p - 1) * p**(e - 1)
    return r
 
 
def divisors(N):
    divs = [1]
    for p, e in factorint(N).items():
        divs = [d * p**k for d in divs for k in range(e + 1)]
    return divs
 
 
def legendre_minus_1_mod_p(p):
    """
    Returns the Kronecker/Legendre symbol (-1/p) for odd prime p.
    """
    return 1 if p % 4 == 1 else -1
 
 
def legendre_minus_3_mod_p(p):
    """
    Returns the Kronecker/Legendre symbol (-3/p) for odd prime p != 3.
    """
    if p == 3:
        return 0
    return 1 if p % 3 == 1 else -1
 
 
def index_gamma0(N):
    """
    μ_0(N) = [SL_2(Z) : Γ_0(N)] = N * ∏_{p|N}(1 + 1/p).
    """
    fac = factorint(N)
    mu0 = N
    for p in fac:
        mu0 *= (1 + 1 / p)
    return int(mu0)
 
 
def elliptic_points_order_2_X0(N):
    """
    Number e_2(N) of elliptic points of order 2 on X_0(N).
    Standard formula:
      e_2(N) = 0 if 4 | N,
      otherwise ∏_{p|N, p odd} (1 + (-1/p)).
    """
    if N % 4 == 0:
        return 0
 
    fac = factorint(N)
    e2 = 1
    for p in fac:
        if p == 2:
            continue
        e2 *= (1 + legendre_minus_1_mod_p(p))
    return e2
 
 
def elliptic_points_order_3_X0(N):
    """
    Number e_3(N) of elliptic points of order 3 on X_0(N).
    Standard formula:
      e_3(N) = 0 if 9 | N or 2 | N,
      otherwise ∏_{p|N, p odd} (1 + (-3/p)).
    """
    if N % 2 == 0 or N % 9 == 0:
        return 0
 
    fac = factorint(N)
    e3 = 1
    for p in fac:
        if p == 2:
            continue
        e3 *= (1 + legendre_minus_3_mod_p(p))
    return e3
 
 
def cusps_X0(N):
    """
    Number of cusps on X_0(N):
      c_0(N) = Σ_{d|N} φ(gcd(d, N/d)).
    """
    return sum(phi(math.gcd(d, N // d)) for d in divisors(N))
 
 
def genus_X0(N):
    """
    Genus of X_0(N):
      g = 1 + μ_0(N)/12 - e_2(N)/4 - e_3(N)/3 - c_0(N)/2.
    """
    mu0 = index_gamma0(N)
    e2 = elliptic_points_order_2_X0(N)
    e3 = elliptic_points_order_3_X0(N)
    c0 = cusps_X0(N)
 
    return int(1 + mu0 / 12 - e2 / 4 - e3 / 3 - c0 / 2)
 
 
def volume_X0(N):
    """
    Hyperbolic volume of X_0(N):
      Vol(X_0(N)) = (π/3) * μ_0(N).
    """
    return (math.pi / 3) * index_gamma0(N)
 
 
def get_X0_data(max_N):
    """
    Returns arrays of genera and volumes for X_0(N), 0 <= N <= max_N.
    Index 0 is padded with 0 for convenience.
    """
    genera = np.zeros(max_N + 1, dtype=int)
    volumes = np.zeros(max_N + 1, dtype=float)
 
    for N in range(1, max_N + 1):
        genera[N] = genus_X0(N)
        volumes[N] = volume_X0(N)
 
    return genera, volumes
 
 
# =============================================================================
#  3  X_0(N) feature matrix
# =============================================================================
 
K = 100
MAX_N = 105
 
X0N_geodesics = get_X0N_geodesics(K, MAX_N)
X0N_genus = np.array([genus_X0(N) for N in range(MAX_N + 1)])
X0N_vol = np.array([0.0] + [volume_X0(N) for N in range(1, MAX_N + 1)], dtype=float)
 
# Restrict to squarefree N
X0N_geodesics = np.array([X0N_geodesics[n] for n in squarefree_Ns], dtype=object)
X0N_genus = np.array([genus_X0(n) for n in squarefree_Ns], dtype=float)
X0N_vol = np.array([volume_X0(n) for n in squarefree_Ns], dtype=float)
 
# Construct feature vectors, where the ith row corresponds to squarefree_Ns[i]
raw_features = []
for i in range(len(squarefree_Ns)):
    # Static features: genus and volume
    static_feats = np.array([X0N_genus[i], X0N_vol[i]], dtype=float)
 
    # Dynamic features: geodesic lengths
    geodesics = np.asarray(X0N_geodesics[i], dtype=float)
 
    combined = np.concatenate((static_feats, geodesics))
    raw_features.append(combined)
 
# Pad to uniform length for MLP input
max_len = max(len(f) for f in raw_features)
padded_features = []
 
for f in raw_features:
    pad_len = max_len - len(f)
    if pad_len > 0:
        f_padded = np.pad(f, (0, pad_len), mode="constant", constant_values=0.0)
    else:
        f_padded = f
    padded_features.append(f_padded)
 
X0N_features = np.vstack(padded_features).astype(np.float32)
print(f"Feature vector constructed. Shape: {X0N_features.shape}")
 
 
# =============================================================================
#  4.  Bolza surface feature vector
# =============================================================================
 
# Bolza surface: first 80 primitive lengths (Aurich & Steiner 1988)
bolza_lengths = np.array([3.0571148390, 4.8969048954, 5.8280707754, 6.1128364779, 6.6720057699, 7.1073578414, 7.2631634751, 7.5956918304, 7.8806928877, 8.1300753289, 8.2249036323, 8.4368496405, 8.6284635656, 8.7027505564, 8.8714798107, 9.0270171797, 9.1714255169, 9.2282950896, 9.3592716579, 9.4821914493, 9.5309770571, 9.6440665486, 9.7510997583, 9.7938097907, 9.8980946367, 10.0785887303,10.1149054144,10.1999558888,10.2815633765,10.3143770353, 10.3915072941,10.4657729697,10.5373792543,10.5663604642,10.6344594159, 10.7308060283,10.7270445783,10.7901078989,10.8510687003,10.8758208811, 10.9343449467,10.9912049969,11.0464930504,11.0689538604,11.1226185852, 11.1739905306,11.2450675676,11.2938444648,11.3416400077,11.3608557221, 11.4060920010,11.4519475300,11.4703055581,11.5139343106,11.5566494092, 11.5984625221,11.6159252909,11.6561415509,11.6954535600,11.7122036393, 11.7509179318,11.7858970234,11.8044196533,11.8410454395,11.8777196407, 11.9133862138,11.9279754971,11.9627646508,11.9969589861,12.0443403136, 12.0771791386,12.1098474621,12.1227186314,12.1543053200,12.1854008904, 12.1981480207,12.2285673558,12.2585379275,12.2708217905])
bolza_genus = 2
bolza_vol = 4 * np.pi
 
# Format: [Genus, Volume, L_1, ..., L_80]
bolza_features = np.zeros(82, dtype=np.float32)
bolza_features[0] = bolza_genus
bolza_features[1] = bolza_vol
num_lengths = len(bolza_lengths)
bolza_features[2 : 2 + num_lengths] = bolza_lengths[:num_lengths]
print(f"Bolza feature vector created. Shape: {bolza_features.shape}")
 
 
# =============================================================================
#  5.  Klein quartic feature vector
# =============================================================================
 
klein_lengths = np.array([3.935962,5.208017,7.358318,7.609408,7.985792,8.205601,8.524421,8.694483,9.040478,9.173079,9.450298,9.643831,9.866002,9.954731,10.026434,10.076370,10.240213,10.345908,10.515278,10.569689,10.632507,10.873258,10.996456,11.047355,11.283088,11.327265,11.390759,11.629027,12.147440,12.503292,12.605421,12.717043,13.467771,13.927559,14.170841,14.281298,14.509264,14.525393,16.758943,17.565838,17.788878,18.418937,18.865298,20.507876,20.562704,21.079984,21.990249,22.083771,22.123374,22.720066,23.379660,24.505599,24.871774,26.800749,27.551624,28.845985,29.063801,30.212253,32.204814,32.57810])
klein_genus = 3
klein_vol = 8 * np.pi
 
klein_features = np.zeros(62, dtype=np.float32)
klein_features[0] = klein_genus
klein_features[1] = klein_vol
num_lengths = len(klein_lengths)
klein_features[2 : 2 + num_lengths] = klein_lengths[:num_lengths]
print(f"Klein feature vector created. Shape: {klein_features.shape}")
