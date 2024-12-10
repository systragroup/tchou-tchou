from tchou_tchou import pso
from shapely import geometry 
import random
import pandas as pd
import math
import numpy as np
from tqdm import tqdm

def relax(geo, tolerance=300):
    coords = list(geo.coords)
    points = [geometry.Point(coords[0])]
    for c in coords[1:]:
        try:
            sofar = geometry.LineString(points)
        except:
            sofar = points[0]
        p = geometry.Point(c)
        s = sofar.distance(p)
        if s >= tolerance:
            points.append(c)
    return geometry.LineString(points)


def get_pos(geo):
    coords = list(geo.coords)
    pos_list = [(x+random.random(), y, 450) for x, y in coords]

    # no R for first HPI
    x, y, r = pos_list[0]
    pos_list[0] = (x, y, 0)
    x, y, r = pos_list[-1]
    pos_list[-1] = (x, y, 0)
    df  = pd.DataFrame(pos_list, columns=['X', 'Y', 'R'])

    return df

def round(geo):

    pos = get_pos(geo).stack().to_frame().T
    rounded = pso._alignment(pos)[0]

    return rounded

def get_segments_dataframe(geo):
    df = get_pos(geo)
    df['s'] = [geo.project(geometry.Point(c)) for c in geo.coords]
    diff = df.diff()
    diff['rad'] = diff.apply(lambda row: math.atan2(row['Y'], row['X']), axis=1)
    diff['drad'] = diff['rad'].diff()
    diff['r'] = diff['s'] /diff['drad']/ np.pi

    df['r'] = diff['r']
    data = df[['X','Y', 'r']].fillna(0).values.tolist()

    segments = []
    for i in range(len(data) - 1):
        xa, ya, ra = data[i]
        xb, yb, rb = data[i + 1]
        t = rb*ra < 0
        segments.append((geometry.LineString([(xa,ya),(xb,yb)]), rb, t))


    sdf = pd.DataFrame(segments, columns=['geometry','r', 't'])
    sdf['inverse_r'] = np.clip(1 / sdf['r'] * 450,-2,2)
    sdf['t'] = sdf['t'].shift(-1)
    
    with pd.option_context("future.no_silent_downcasting", True):
        sdf['t'] = sdf['t'].fillna(False)

    sdf['drad'] = diff['drad'].shift(-1)
    sdf['length'] = diff['s'].shift(-1)
    sdf['r'] = sdf['r'].shift(-1)
    sdf['abs_r'] = np.abs(sdf['r'])

    start, end = list(sdf.index)[0], list(sdf.index)[-1]
    sdf.loc[start, 't'] = True
    sdf.loc[end, 't'] = True
    


    return sdf


def get_xy_crossing(segments):
    
    line = list(segments.iloc[0]['geometry'].coords)
    xa, ya = line[0]
    xb, yb = line[1]
    alpha = (yb - ya) / (xb - xa)
    beta = ya - alpha * xa
    alpha_o, beta_o = alpha, beta

    line = list(segments.iloc[-1]['geometry'].coords)
    xa, ya = line[0]
    xb, yb = line[1]
    alpha = (yb - ya) / (xb - xa)
    beta = ya - alpha * xa
    alpha_d, beta_d = alpha, beta

    x = (beta_d - beta_o) / (alpha_o - alpha_d)
    y = alpha * x + beta
    return x, y

def get_geometry(curve, dx=0, dy=0, r=450):
    curve_df = pd.DataFrame(curve, columns=['X', 'Y'])
    curve_df['R'] = 0
    curve_df.loc[1, 'R'] = r
    curve_df.loc[1, 'X'] += dx
    curve_df.loc[1, 'Y'] += dy
    curve_pos = curve_df.stack().to_frame().T
    return pso._alignment(curve_pos)[0]


def distance(ga, gb):
    for b in range(0, 10000, 10):
        if ga.buffer(b).contains(gb):
            break

    for b_fine in range(b-10, b+10, 1):
        if ga.buffer(b_fine).contains(gb):
            return b_fine
        
def get_dx_dy_r(curve, lines, start_radius=1000, tolerance=100, maxiter=30, step_xy=5, step_r=50):
    dx, dy, r = 0, 0, start_radius

    for i in list(range(maxiter)):
        curve_geometry = get_geometry(curve, dx, dy, r)
        dist = distance(lines, curve_geometry)
        #print(dx, dy, r, dist)

        # SCAN
        best_ddx, best_ddy, best_dr = 0, 0, 0

        best = dist
        for ddx in [-step_xy, 0, step_xy]:
            for ddy in [-step_xy, 0, step_xy]:
                for dr in [-step_r, 0, step_r]:
                    challenger_geometry = get_geometry(curve, dx+ddx, dy+ddy, r+dr)
                    challenger = distance(lines, challenger_geometry)
                    if challenger < best:
                        best = challenger
                        best_ddx, best_ddy, best_dr = ddx, ddy, dr
                    

        if best_ddx == 0 and best_ddy == 0 and best_dr == 0:
            step_xy = int(step_xy*0.5)
            step_r = int(step_r*0.5)
                        
        dx += best_ddx
        dy += best_ddy
        r += best_dr

        if best < tolerance:
            break
    #print(i, print(dx + ddx, dy+ddy, r+dr))
    return dx+ddx, dy+ddy, r+dr

def add_tangents_in_curves(segments, max_length=2000):
    df = segments.copy()

    df['segment'] = df['t'].astype(int).cumsum()

    analyst = df.groupby('segment').agg({'length': 'sum', 'abs_r': 'max'})

    last = df.groupby('segment').first()['length'].shift(-1) / 2
    first = df.groupby('segment').last()['length'].shift(1) / 2
    analyst['length'] = analyst['length'] - first.fillna(0) + last.fillna(0)

    max_r = df.groupby('segment')['abs_r'].max()
    first = df.groupby('segment')['abs_r'].first()
    last = df.groupby('segment')['abs_r'].last()
    right = pd.DataFrame({'max': max_r, 'first': first, 'last': last}).fillna(0)
    flat = right.loc[(right['max'] > right['first']) & (right['max'] > right['last'])]
    flat = pd.merge(analyst, flat, left_index=True, right_index=True, how='left').dropna()

    new_t = flat.loc[flat['length'] > max_length]['abs_r'].reset_index()
    new_t['new_t'] = True
    df = pd.merge(df, new_t, on=['segment', 'abs_r'], how='left')
    df.loc[df['new_t'] == True, 't'] = True

    return df[segments.columns]

def fit_pos(segments, start_radius=750, maxiter=30, tolerance=30, step_xy=10):

    df = segments.copy()
    ix = list(df.loc[df['t']].index)
    slices = [(ix[i], ix[i+1]) for i in range(len(ix)-1)]

    results = {}

    for slc in tqdm(slices):
        seg_slice = df.loc[slc[0]:slc[1]]
        lines = geometry.MultiLineString(list(seg_slice['geometry']))
        lo = list(seg_slice.iloc[0]['geometry'].coords)
        ld = list(seg_slice.iloc[-1]['geometry'].coords)

        x_i, y_i = get_xy_crossing(segments=seg_slice)
        curve = [lo[0], (x_i, y_i), ld[1]]
        dx, dy, r = get_dx_dy_r(curve, lines, start_radius=start_radius, maxiter=maxiter, tolerance=tolerance, step_xy=step_xy)

        lines.buffer(50) - get_geometry(curve, dx, dy, r).buffer(1)
        results[slc] = [curve, dx, dy, r]


    data = []
    for curve,  dx, dy, r in  results.values():
        x = curve[1][0] + dx
        y = curve[1][1] + dy
        data.append([x, y, r])

    xo, yo = list(df.iloc[0]['geometry'].coords)[0]
    xd, yd = list(df.iloc[-1]['geometry'].coords)[-1]

    data = [[xo, yo, 0]] + data + [[xd, yd, 0]]

    return pd.DataFrame(data, columns=['X', 'Y', 'R'])


def get_tangents(pos):
    "return companion dataframe of POS focusing on tangents pos has ['X', 'Y', 'R'] columns"

    geo = pso._alignment(pos.stack().to_frame().T)[0]
    fix = pos.copy()

    left = pd.DataFrame(list(geo.coords)[:-1], columns=['xa', 'ya'])
    right = pd.DataFrame(list(geo.coords)[1:], columns=['xb', 'yb'])
    df = pd.concat([left, right], axis=1)

    def dist(xa, ya, xb, yb):
        return np.sqrt(np.power(xb-xa, 2) + np.power(yb-ya, 2))
    df['length'] = dist(df['xa'], df['ya'], df['xb'], df['yb'])

    tangents = df.loc[df['length']> 50].copy()
    tangents['geometry'] = tangents.apply(lambda r: geometry.LineString([[r['xa'], r['ya']], [r['xb'], r['yb']]]), axis=1)

    tangents['s'] = tangents['geometry'].apply(lambda g: geo.project(geometry.Point(list(g.coords)[0])))
    fix['s'] = fix.apply(lambda r: geo.project(geometry.Point(r['X'], r['Y'])), axis=1)
    ix_cs = fix['s'].reset_index().values.tolist()

    def get_before(ts):
        for before_ix, cs in ix_cs:
            if cs > ts :
                return before_ix
            
    tangents['curve'] = tangents['s'].apply(get_before) -1
    tangents['dy'] = (tangents['yb'] - tangents['ya']) / tangents['length']
    tangents['dx'] = (tangents['xb'] - tangents['xa']) / tangents['length']
    
    fix['front_tan'] = tangents.groupby('curve')['length'].max()
    fix.loc[fix.index[-1], 'front_tan'] = 1000
    fix = fix.fillna(0)#.iloc[1:]

    return fix, tangents.reset_index(drop=True)

def contract_radius(pos, tangents=None, tan_min=300, alpha=0.75):
    if tangents is None :
        tangents = get_tangents(pos)[1]
    tangents = tangents.copy()
    fix = pos.copy()
    
    tangents['push'] = (tan_min - tangents['length'] ) * alpha
    push = tangents.loc[tangents['push'] > 0][['curve', 'dx', 'dy', 'push']].values
    for back_curve, dx, dy, p in push:
        fix.loc[back_curve, 'R'] -= p 
        fix.loc[back_curve + 1, 'R'] -= p 

    return fix

def de_constrain(pos, tan_min):
    fix = get_tangents(pos)
    fix['front_short'] = (fix['front_tan'] < tan_min)
    fix['back_short'] = fix['front_short'].shift(1)
    to_drop = fix.loc[fix['front_short'] & fix['back_short']].index
    return fix.drop(to_drop)

def push_curves(pos, tangents=None, tan_min=300, alpha=0.75):
    if tangents is None :
        tangents = get_tangents(pos)[1]
    tangents = tangents.copy()
    fix = pos.copy()
    
    tangents['push'] = (tan_min - tangents['length'] ) * alpha
    push = tangents.loc[tangents['push'] > 0][['curve', 'dx', 'dy', 'push']].values
    for back_curve, dx, dy, p in push:
        fix.loc[back_curve, 'X'] -= p * dx
        fix.loc[back_curve + 1, 'X'] += p* dx
        fix.loc[back_curve, 'Y'] -= p * dy
        fix.loc[back_curve + 1, 'Y'] += p * dy

    return fix