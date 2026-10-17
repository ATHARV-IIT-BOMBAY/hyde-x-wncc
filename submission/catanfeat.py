"""Compact, observation-safe features of a Catan position from one player's perspective.

Only information the player can see is used: our own full hand (incl. dev cards / hidden VPs),
the opponent's public VPs, resource hand (derivable from bank) and dev-card COUNT.
"""
from catanatron.models.board import STATIC_GRAPH
from catanatron.models.enums import RESOURCES, SETTLEMENT, CITY, ROAD
from catanatron.models.map import number_probability

NEIGHBORS = {n: tuple(STATIC_GRAPH.neighbors(n)) for n in STATIC_GRAPH.nodes}
DEVS = ["KNIGHT", "VICTORY_POINT", "ROAD_BUILDING", "YEAR_OF_PLENTY", "MONOPOLY"]


class MapInfo:
    def __init__(self, cmap):
        self.cmap = cmap
        coord_of = {id(t): c for c, t in cmap.land_tiles.items()}
        self.node_tiles = {}
        self.node_prod_total = {}
        for n, tiles in cmap.adjacent_tiles.items():
            lst = [(coord_of.get(id(t)), t.resource, number_probability(t.number))
                   for t in tiles if t.resource is not None]
            self.node_tiles[n] = lst
            self.node_prod_total[n] = sum(p for _, _, p in lst)
        # node -> port resource (None = 3:1); absent = no port
        self.node_port = {}
        for res, nodes in cmap.port_nodes.items():
            for n in nodes:
                self.node_port[n] = res


_MI = [None]


def mapinfo(cmap):
    mi = _MI[0]
    if mi is None or mi.cmap is not cmap:
        mi = MapInfo(cmap)
        _MI[0] = mi
    return mi


def side_features(state, color, other, mi, robber, out):
    bbc = state.buildings_by_color[color]
    setts = bbc[SETTLEMENT]
    cities = bbc[CITY]
    prod = dict.fromkeys(RESOURCES, 0.0)
    nt = mi.node_tiles
    for n in setts:
        for c, r, p in nt[n]:
            if c != robber:
                prod[r] += p
    for n in cities:
        for c, r, p in nt[n]:
            if c != robber:
                prod[r] += 2 * p
    tot = 0.0
    ntypes = 0
    for r in RESOURCES:
        v = prod[r]
        out.append(v)
        tot += v
        ntypes += v > 0
    out.append(tot)
    out.append(ntypes)
    out.append(len(setts))
    out.append(len(cities))
    out.append(len(bbc[ROAD]))
    # ports
    has3 = 0
    port2 = set()
    for n in setts + cities:
        pr = mi.node_port.get(n, 0)
        if pr is None:
            has3 = 1
        elif pr != 0:
            port2.add(pr)
    out.append(has3)
    out.append(len(port2))
    out.append(sum(prod[r] for r in port2))
    # reachability: buildable connected nodes and production reachable within one road
    board = state.board
    zero = set()
    for comp in board.connected_components[color]:
        zero |= comp
    buildable = zero & board.board_buildable_ids
    out.append(len(buildable))
    npt = mi.node_prod_total
    out.append(sum(npt.get(n, 0.0) for n in buildable))
    enemy_nodes = set(state.buildings_by_color[other][SETTLEMENT]) | set(
        state.buildings_by_color[other][CITY])
    roads = board.roads
    level = set()
    for n in zero:
        if n in enemy_nodes:
            continue
        for nb in NEIGHBORS[n]:
            if nb in zero:
                continue
            rc = roads.get((n, nb))
            if rc is None:
                level.add(nb)
    reach = level & board.board_buildable_ids
    out.append(len(reach))
    out.append(max([npt.get(n, 0.0) for n in reach], default=0.0))
    return prod


def features(state, me, me_to_move_next=0):
    """Feature vector (list of floats) of `state` from `me`'s perspective."""
    ps = state.player_state
    i = state.color_to_index[me]
    j = 1 - i
    opp = state.colors[j]
    P = f"P{i}_"
    Q = f"P{j}_"
    mi = mapinfo(state.board.map)
    robber = state.board.robber_coordinate
    out = []
    # victory points
    my_vp = ps[P + "ACTUAL_VICTORY_POINTS"]
    opp_vp = ps[Q + "VICTORY_POINTS"]
    out += [my_vp, opp_vp, my_vp - opp_vp]
    side_features(state, me, opp, mi, robber, out)
    side_features(state, opp, me, mi, robber, out)
    # roads / army
    out += [ps[P + "LONGEST_ROAD_LENGTH"], ps[Q + "LONGEST_ROAD_LENGTH"],
            ps[P + "HAS_ROAD"], ps[Q + "HAS_ROAD"],
            ps[P + "PLAYED_KNIGHT"], ps[Q + "PLAYED_KNIGHT"],
            ps[P + "HAS_ARMY"], ps[Q + "HAS_ARMY"]]
    # hands
    mh = [ps[P + r + "_IN_HAND"] for r in RESOURCES]
    oh = [ps[Q + r + "_IN_HAND"] for r in RESOURCES]
    out += mh
    out += oh
    nm = sum(mh)
    no = sum(oh)
    out += [nm, no, float(nm > 7), float(no > 7)]
    wood, brick, sheep, wheat, ore = mh
    out.append((min(wheat, 2) + min(ore, 3)) / 5.0)                       # city readiness
    out.append((min(wood, 1) + min(brick, 1) + min(sheep, 1) + min(wheat, 1)) / 4.0)
    out.append((min(sheep, 1) + min(wheat, 1) + min(ore, 1)) / 3.0)       # dev readiness
    # dev cards
    out += [ps[P + d + "_IN_HAND"] for d in DEVS]
    out.append(sum(ps[Q + d + "_IN_HAND"] for d in DEVS))
    out.append(len(state.development_listdeck))
    # tempo
    out.append(min(state.num_turns, 200) / 100.0)
    out.append(me_to_move_next)
    return out


N_FEATURES = None


def n_features():
    return 3 + 2 * 18 + 8 + 10 + 4 + 3 + 5 + 2 + 2
