import numpy as np
import math

#######################
# Function definition #
#######################

# v_*: vectorialized form of the function

def compute_rs(x, y, H, r_eq, r_pol):
	a = pow(math.sin(x),2) + pow(math.cos(x),2)*(pow(math.cos(y),2) + pow((r_eq/r_pol)*math.sin(y),2))
	b = -2*H*math.cos(x)*math.cos(y)
	c = pow(H,2) - pow(r_eq,2)
	num = 0
	if (pow(b,2)-(4*a*c)) >= 0: num = -b - math.sqrt(pow(b,2)-(4*a*c))
	den = 2*a
	return num/den
v_compute_rs = np.vectorize(compute_rs)

def compute_sx_1(rs, x, y): return rs*math.cos(x)*math.cos(y)
v_compute_sx_1 = np.vectorize(compute_sx_1)

def compute_sy_1(rs, x): return -rs*math.sin(x)
v_compute_sy_1 = np.vectorize(compute_sy_1)

def compute_sz_1(rs, x, y): return rs*math.cos(x)*math.sin(y)
v_compute_sz_1 = np.vectorize(compute_sz_1)

def compute_lat(sx, sy, sz, H, r_eq, r_pol): return math.atan(pow(r_eq/r_pol,2)*sz/math.sqrt(pow(H-sx,2)+pow(sy,2)))
v_compute_lat = np.vectorize(compute_lat)

def compute_lon(sx, sy, H, lambda_0): return lambda_0 - math.atan(sy/(H-sx))
v_compute_lon = np.vectorize(compute_lon)

def compute_phi_c(r_pol, r_eq, lat): return math.atan(pow(r_pol/r_eq,2)*math.tan(lat))
v_compute_phi_c = np.vectorize(compute_phi_c)

def compute_rc(r_pol, e, phi_c): return r_pol/math.sqrt(1-pow(e*math.cos(phi_c),2))
v_compute_rc = np.vectorize(compute_rc)

def compute_sx_2(H, rc, phi_c, lon, lambda_0): return H - rc*math.cos(phi_c)*math.cos(lon-lambda_0)
v_compute_sx_2 = np.vectorize(compute_sx_2)

def compute_sy_2(rc, phi_c, lon, lambda_0): return -rc*math.cos(phi_c)*math.sin(lon-lambda_0)
v_compute_sy_2 = np.vectorize(compute_sy_2)

def compute_sz_2(rc, phi_c): return rc*math.sin(phi_c)
v_compute_sz_2 = np.vectorize(compute_sz_2)

def compute_x(sx, sy, sz):
	num = -sy
	den = math.sqrt(pow(sx,2)+pow(sy,2)+pow(sz,2))
	return math.asin(num/den)
v_compute_x = np.vectorize(compute_x)

def compute_y(sx, sz): return math.atan(sz/sx)
v_compute_y = np.vectorize(compute_y)

def compute_x_sin(lat, lon, lambda_0): return (lon-lambda_0)*math.cos(lat)
v_compute_x_sin = np.vectorize(compute_x_sin)

def compute_lon_sin(x, lat, lambda_0):
	lon = (x/math.cos(lat)) + lambda_0
	#if lon > math.pi: return 2*math.pi - lon
	#elif lon < -math.pi: return lon + 2*math.pi
	#else: return lon
	return lon
v_compute_lon_sin = np.vectorize(compute_lon_sin)

def compute_lza(H, lat, lon, lambda_0, r_eq):
	beta = math.acos(math.cos(lat)*math.cos(lon-lambda_0))
	num = H*math.sin(beta)
	den = math.sqrt(pow(H,2) + pow(r_eq,2) - 2*H*r_eq*math.cos(beta))
	return math.asin(num/den)
v_compute_lza = np.vectorize(compute_lza)
