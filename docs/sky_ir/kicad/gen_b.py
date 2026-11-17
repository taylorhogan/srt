"""Part B: footprints, board, project file."""
import uuid, json, os, sys
OUT = sys.argv[1]
S = json.load(open(f"{OUT}/.gen_state.json"))
ROW, ROOT, SYMU, CONN, NETS, VALUES, FOOTS, DATE = (S[k] for k in ["ROW","ROOT","SYMU","CONN","NETS","VALUES","FOOTS","DATE"])
def U(): return str(uuid.uuid4())
NETNO = {n: i+1 for i, n in enumerate(NETS)}
NAMES44 = S["J1_NAMES"] + S["J3_NAMES"]
for i in range(44):
    num = str(i+1)
    if num not in CONN["U1"]:
        nm = NAMES44[i]; net = f"unconnected-(U1-Pad{num})" if nm == num else f"unconnected-(U1-{nm}-Pad{num})"
        CONN["U1"][num] = net; NETNO[net] = len(NETNO) + 1
W = round(40 + ROW + 8, 2); H = 72.0
FE = '(effects (font (size 1 1) (thickness 0.15)))'

# ---------------- footprints: pads relative to the footprint origin ----------------
def pad(num, shape, x, y, size, drill, net=None):
    n = f' (net {NETNO[net]} "{net}")' if net else ""
    return f'(pad "{num}" thru_hole {shape} (at {x} {y}) (size {size} {size}) (drill {drill}) (layers "*.Cu" "*.Mask") (remove_unused_layers no){n} (uuid "{U()}"))'
def npth(x, y, d):
    return f'(pad "" np_thru_hole circle (at {x} {y}) (size {d} {d}) (drill {d}) (layers "*.Cu" "*.Mask") (uuid "{U()}"))'
def line(a, b, layer="F.SilkS", w=0.12):
    return f'(fp_line (start {a[0]} {a[1]}) (end {b[0]} {b[1]}) (stroke (width {w}) (type solid)) (layer "{layer}") (uuid "{U()}"))'
def rect(x0, y0, x1, y1, layer="F.SilkS", w=0.12):
    return [line((x0,y0),(x1,y0),layer,w), line((x1,y0),(x1,y1),layer,w), line((x1,y1),(x0,y1),layer,w), line((x0,y1),(x0,y0),layer,w)]

def fp_header(n, nets):            # 1x n, 2.54 pitch, pin 1 at origin, going +y
    g = [pad("1","rect",0,0,1.7,1.0,nets.get("1"))] + [pad(str(i+1),"circle",0,round(i*2.54,2),1.7,1.0,nets.get(str(i+1))) for i in range(1,n)]
    g += rect(-1.27,-1.27,1.27,(n-1)*2.54+1.27) + rect(-1.27,-1.27,1.27,(n-1)*2.54+1.27,"F.CrtYd",0.05) + [line((-1.27,1.27),(1.27,1.27))]
    return g, [(0, i*2.54) for i in range(n)]
def fp_esp(nets):                  # two 1x22 sockets, pin 1 of J1 at origin, J3 column at x=ROW
    g = []; pads = []
    for col, base in [(0,1),(ROW,23)]:
        for i in range(22):
            num = str(base+i); y = round(i*2.54,2)
            g.append(pad(num, "rect" if i==0 else "circle", col, y, 1.7, 1.0, nets.get(num))); pads.append((col,y))
        g += rect(col-1.27,-1.27,col+1.27,21*2.54+1.27)
    g += rect(-1.27,-6.35,ROW+1.27,21*2.54+6.35,"F.Fab",0.1) + rect(-1.27,-6.35,ROW+1.27,21*2.54+6.35,"F.CrtYd",0.05)
    g.append(f'(fp_text user "USB-C this end" (at {ROW/2:.2f} {21*2.54+4.5:.2f} 0) (layer "F.SilkS") (uuid "{U()}") {FE})')
    g.append(f'(fp_text user "3V3  3V3" (at -4.5 0.6 90) (layer "F.SilkS") (uuid "{U()}") {FE})')
    return g, pads
def fp_axial(nets):                # 10.16 pitch horizontal, pin 1 at origin
    g = [pad("1","circle",0,0,1.8,0.9,nets.get("1")), pad("2","circle",10.16,0,1.8,0.9,nets.get("2"))]
    g += rect(1.9,-1.3,8.26,1.3) + [line((1.27,0),(1.9,0)), line((8.26,0),(8.89,0))] + rect(-1.3,-1.6,11.46,1.6,"F.CrtYd",0.05)
    return g, [(0,0),(10.16,0)]
def fp_to220(nets):                # G D S at 2.54 pitch, pin 1 at origin, body behind (+y is away from the board edge)
    g = [pad("1","rect",0,0,1.9,1.1,nets.get("1")), pad("2","circle",2.54,0,1.9,1.1,nets.get("2")), pad("3","circle",5.08,0,1.9,1.1,nets.get("3"))]
    g += rect(-2.54,1.3,7.62,3.8) + rect(-2.8,-1.3,7.88,4.1,"F.CrtYd",0.05)
    g.append(f'(fp_text user "G D S" (at 2.54 -2.2 0) (layer "F.SilkS") (uuid "{U()}") {FE})')
    return g, [(0,0),(2.54,0),(5.08,0)]
def fp_hole():
    return [npth(0,0,3.2), f'(fp_circle (center 0 0) (end 3.0 0) (stroke (width 0.12) (type solid)) (fill none) (layer "F.SilkS") (uuid "{U()}"))'], []

# ---------------- placement (board coordinates, mm, y down) ----------------
PLACE = {   # ref: (footprint name, x, y, builder)
 "U1": ("ESP32-S3-DevKitC-1_Sockets", 40, 10, lambda: fp_esp(CONN["U1"])),
 "J2": ("PinHeader_1x04", 8, 14, lambda: fp_header(4, CONN["J2"])),
 "J4": ("PinHeader_1x04", 8, 29, lambda: fp_header(4, CONN["J4"])),
 "J5": ("PinHeader_1x02", 8, 52, lambda: fp_header(2, CONN["J5"])),
 "R1": ("R_Axial_P10.16mm", 20, 12.54, lambda: fp_axial(CONN["R1"])),   # pin1 at (20,12.54) ... wait: vertical needed
 "R2": ("R_Axial_P10.16mm", 24, 12.54, lambda: fp_axial(CONN["R2"])),
 "R3": ("R_Axial_P10.16mm", 30.16, 48.1, lambda: fp_axial(CONN["R3"])),
 "R4": ("R_Axial_P10.16mm", 20, 55, lambda: fp_axial(CONN["R4"])),
 "Q1": ("TO-220-3_Vertical", 20, 62, lambda: fp_to220(CONN["Q1"])),
}
ROT = {"R1": 270, "R2": 270, "R3": 180}      # R1/R2 stand vertical (pin 2 at +y); R3 pin 1 on the right
REFPOS = {"R1": (5.08,2.6), "R2": (5.08,-2.6), "R3": (5.08,-2.6), "R4": (5.08,2.6), "Q1": (0,5.3)}
HOLES = [(4,4),(W-4,4),(4,H-4),(W-4,H-4)]

def rot(x, y, a):
    import math
    r = math.radians(a); c, s = round(math.cos(r)), round(math.sin(r))
    return (round(x*c - y*s, 2), round(x*s + y*c, 2))

fps = []; PADS = {}; MODS = {}
for ref, (name, x, y, build) in PLACE.items():
    g, pads = build(); a = ROT.get(ref, 0)
    # KiCad: footprint (at x y a) rotates children by a (counter-clockwise on screen because y is down... KiCad rotates pads by -a in file coords)
    PADS[ref] = [(round(x + rot(px, py, -a)[0], 2), round(y + rot(px, py, -a)[1], 2)) for px, py in pads]
    import re as _re
    if a:
        g = [_re.sub(r'\(pad ("[^"]*") (thru_hole|np_thru_hole) (\w+) \(at ([-\d.]+) ([-\d.]+)\)', lambda m: f'(pad {m.group(1)} {m.group(2)} {m.group(3)} (at {m.group(4)} {m.group(5)} {a})', s) for s in g]
    body = "\n    ".join(g)
    hdr = f'''(footprint "sky_ir:{name}" (layer "F.Cu") (uuid "{U()}") (at {x} {y} {a})
    (property "Reference" "{ref}" (at {REFPOS.get(ref,(0,-2.4))[0]} {REFPOS.get(ref,(0,-2.4))[1]} {a}) (unlocked yes) (layer "F.SilkS") (uuid "{U()}") {FE})
    (property "Value" "{VALUES[ref]}" (at 0 2.4 {a}) (unlocked yes) (layer "F.Fab") (uuid "{U()}") {FE})
    (property "Footprint" "sky_ir:{name}" (at 0 0 0) (unlocked yes) (layer "F.Fab") hide (uuid "{U()}") {FE})
    (property "Datasheet" "" (at 0 0 0) (unlocked yes) (layer "F.Fab") hide (uuid "{U()}") {FE})
    (property "Description" "" (at 0 0 0) (unlocked yes) (layer "F.Fab") hide (uuid "{U()}") {FE})
    (path "/{SYMU[ref]}") (sheetname "/") (sheetfile "sky_ir.kicad_sch")
    (attr through_hole)
    {body})'''
    fps.append(hdr)
    if name not in MODS:
        g2, _ = build()
        import re
        clean = re.sub(r' \(net \d+ "[^"]*"\)', '', "\n  ".join(g2))
        MODS[name] = f'''(footprint "{name}" (version 20240108) (generator "pcbnew") (generator_version "8.0") (layer "F.Cu")
  (property "Reference" "REF**" (at 0 -2.4 0) (unlocked yes) (layer "F.SilkS") (uuid "{U()}") {FE})
  (property "Value" "{name}" (at 0 2.4 0) (unlocked yes) (layer "F.Fab") (uuid "{U()}") {FE})
  (attr through_hole)
  {clean})'''
for i, (hx, hy) in enumerate(HOLES):
    g, _ = fp_hole()
    fps.append(f'''(footprint "sky_ir:MountingHole_3.2mm" (layer "F.Cu") (uuid "{U()}") (at {hx:.2f} {hy:.2f} 0)
    (property "Reference" "H{i+1}" (at 0 -4 0) (unlocked yes) (layer "F.SilkS") hide (uuid "{U()}") {FE})
    (property "Value" "M3" (at 0 4 0) (unlocked yes) (layer "F.Fab") (uuid "{U()}") {FE})
    (attr through_hole board_only exclude_from_bom)
    {chr(10).join(g)})''')
g, _ = fp_hole()
MODS["MountingHole_3.2mm"] = f'(footprint "MountingHole_3.2mm" (version 20240108) (generator "pcbnew") (generator_version "8.0") (layer "F.Cu")\n  (property "Reference" "REF**" (at 0 -4 0) (unlocked yes) (layer "F.SilkS") hide (uuid "{U()}") {FE})\n  (property "Value" "MountingHole_3.2mm" (at 0 4 0) (unlocked yes) (layer "F.Fab") (uuid "{U()}") {FE})\n  (attr through_hole exclude_from_bom)\n  {chr(10).join(g)})'

# ---------------- routing ----------------
P = PADS
j1 = lambda n: P["U1"][n-1]
tracks = []; vias = []
def seg(net, pts, layer, w=0.5):
    for a, b in zip(pts, pts[1:]): tracks.append((net, a, b, layer, w))
def via(net, p): vias.append((net, p))
# +3V3: F.Cu line y=12.54 from J1.2 across R2.1, R1.1 to a via at x=16; B.Cu bus x=16 to the headers
seg("+3V3", [j1(1), j1(2)], "F.Cu"); seg("+3V3", [j1(2), (16,12.54)], "F.Cu"); via("+3V3", (16,12.54))
seg("+3V3", [(16,12.54),(16,14)], "B.Cu"); via("+3V3",(16,14)); seg("+3V3", [P["J2"][0],(16,14)], "F.Cu")
seg("+3V3", [(16,14),(16,29)], "B.Cu"); via("+3V3",(16,29)); seg("+3V3", [P["J4"][0],(16,29)], "F.Cu")
# SDA: J2.4 (8,21.62) and J4.4 (8,36.62) stubs to vias at x=12; bus on B.Cu; R1.2 joins at y=24; up to J1.12 on F.Cu at y=37.94
seg("SDA", [P["J2"][3],(12,21.62)], "F.Cu"); via("SDA",(12,21.62))
seg("SDA", [P["R1"][1],(20,24),(12,24)], "F.Cu"); via("SDA",(12,24))
seg("SDA", [P["J4"][3],(12,36.62)], "F.Cu"); via("SDA",(12,36.62))
seg("SDA", [(12,21.62),(12,24),(12,36.62),(12,37.94)], "B.Cu"); via("SDA",(12,37.94))
seg("SDA", [(12,37.94), j1(12)], "F.Cu")
# SCL: stubs to x=14; R2.2 joins at y=26; bus on B.Cu down to y=45.56 then across on B.Cu to J1.15
seg("SCL", [P["J2"][2],(14,19.08)], "F.Cu"); via("SCL",(14,19.08))
seg("SCL", [P["R2"][1],(24,26),(14,26)], "F.Cu"); via("SCL",(14,26))
seg("SCL", [P["J4"][2],(14,34.08)], "F.Cu"); via("SCL",(14,34.08))
seg("SCL", [(14,19.08),(14,26),(14,34.08),(14,45.56), j1(15)], "B.Cu")
# GPIO10 -> R3.1 (right end); GATE: R3.2 (20,48.1) down to Q1.G (20,62) through R4.1 (20,55)
seg("GPIO10", [j1(16), P["R3"][0]], "F.Cu")
seg("GATE", [P["R3"][1], P["R4"][0], P["Q1"][0]], "F.Cu")
# +5V: J1.21 to J5.1 on B.Cu (clear of the buses, which end above y=46)
seg("+5V", [j1(21),(34,60.8),(34,52),P["J5"][0]], "B.Cu", 0.8)
# DRAIN: J5.2 under the MOSFET row to Q1.D from below
seg("DRAIN", [P["J5"][1],(14,54.54),(14,66),(22.54,66),P["Q1"][1]], "F.Cu", 0.8)
# GND: B.Cu pour + a stitch from Q1.S to J1.22 so the source does not depend on the pour alone
seg("GND", [P["Q1"][2],(25.08,63.34), j1(22)], "F.Cu", 0.8)

segs = "\n  ".join(f'(segment (start {a[0]} {a[1]}) (end {b[0]} {b[1]}) (width {w}) (layer "{L}") (net {NETNO[n]}) (uuid "{U()}"))' for n,a,b,L,w in tracks)
vs = "\n  ".join(f'(via (at {p[0]} {p[1]}) (size 0.8) (drill 0.4) (layers "F.Cu" "B.Cu") (net {NETNO[n]}) (uuid "{U()}"))' for n,p in vias)
nets = "\n  ".join(f'(net {i} "{n}")' for n,i in NETNO.items())
pcb = f'''(kicad_pcb (version 20240108) (generator "pcbnew") (generator_version "8.0")
  (general (thickness 1.6) (legacy_teardrops no))
  (paper "A4")
  (title_block (title "Sky IR sensor carrier") (date "{DATE}") (rev "1") (company "Iris observatory"))
  (layers
    (0 "F.Cu" signal) (31 "B.Cu" signal)
    (32 "B.Adhes" user "B.Adhesive") (33 "F.Adhes" user "F.Adhesive") (34 "B.Paste" user) (35 "F.Paste" user)
    (36 "B.SilkS" user "B.Silkscreen") (37 "F.SilkS" user "F.Silkscreen") (38 "B.Mask" user) (39 "F.Mask" user)
    (40 "Dwgs.User" user "User.Drawings") (41 "Cmts.User" user "User.Comments") (42 "Eco1.User" user "User.Eco1") (43 "Eco2.User" user "User.Eco2")
    (44 "Edge.Cuts" user) (45 "Margin" user) (46 "B.CrtYd" user "B.Courtyard") (47 "F.CrtYd" user "F.Courtyard") (48 "B.Fab" user) (49 "F.Fab" user)
  )
  (setup (pad_to_mask_clearance 0) (allow_soldermask_bridges_in_footprints no))
  (net 0 "")
  {nets}
  {chr(10).join(fps)}
  (gr_rect (start 0 0) (end {W} {H}) (stroke (width 0.1) (type default)) (fill none) (layer "Edge.Cuts") (uuid "{U()}"))
  (gr_text "SKY IR  Iris  {DATE[:7]}" (at {W/2:.2f} 3 0) (layer "F.SilkS") (uuid "{U()}") (effects (font (size 1.5 1.5) (thickness 0.3))))
  (gr_text "MLX" (at 4 10 90) (layer "F.SilkS") (uuid "{U()}") {FE})
  (gr_text "BME" (at 4 25 90) (layer "F.SilkS") (uuid "{U()}") {FE})
  (gr_text "HEATER" (at 4 58 90) (layer "F.SilkS") (uuid "{U()}") {FE})
  (gr_text "VIN GND SCL SDA" (at 3 42 90) (layer "F.SilkS") (uuid "{U()}") {FE})
  (gr_text "GND pour on the back" (at {W/2:.2f} 3 0) (layer "B.SilkS") (uuid "{U()}") (effects (font (size 1 1) (thickness 0.15)) (justify mirror)))
  {segs}
  {vs}
  (zone (net {NETNO["GND"]}) (net_name "GND") (layer "B.Cu") (uuid "{U()}") (hatch edge 0.5)
    (connect_pads (clearance 0.5)) (min_thickness 0.25) (filled_areas_thickness no)
    (fill yes (thermal_gap 0.5) (thermal_bridge_width 0.5))
    (polygon (pts (xy 0 0) (xy {W} 0) (xy {W} {H}) (xy 0 {H}))))
)
'''
open(f"{OUT}/sky_ir.kicad_pcb","w").write(pcb)
os.makedirs(f"{OUT}/sky_ir.pretty", exist_ok=True)
for name, txt in MODS.items(): open(f"{OUT}/sky_ir.pretty/{name}.kicad_mod","w").write(txt + "\n")
open(f"{OUT}/fp-lib-table","w").write('(fp_lib_table (version 7)\n  (lib (name "sky_ir")(type "KiCad")(uri "${KIPRJMOD}/sky_ir.pretty")(options "")(descr "Sky IR sensor footprints"))\n)\n')
pro = {"meta": {"filename": "sky_ir.kicad_pro", "version": 1},
       "board": {"design_settings": {"rules": {"min_clearance": 0.2, "min_track_width": 0.25, "min_via_diameter": 0.6, "min_via_hole": 0.3}}},
       "net_settings": {"classes": [{"name": "Default", "clearance": 0.2, "track_width": 0.5, "via_diameter": 0.8, "via_drill": 0.4,
                        "diff_pair_width": 0.2, "diff_pair_gap": 0.25, "diff_pair_via_gap": 0.25, "microvia_diameter": 0.3, "microvia_drill": 0.1,
                        "wire_width": 6, "bus_width": 12, "line_style": 0, "pcb_color": "rgba(0, 0, 0, 0.000)", "schematic_color": "rgba(0, 0, 0, 0.000)"}],
                        "meta": {"version": 3}},
       "schematic": {"legacy_lib_dir": "", "legacy_lib_list": []}, "sheets": [[ROOT, "Root"]], "text_variables": {}}
json.dump(pro, open(f"{OUT}/sky_ir.kicad_pro","w"), indent=2)
json.dump({"PADS": PADS, "tracks": tracks, "vias": vias, "W": W, "H": H, "ROW": ROW, "ROT": ROT, "PLACE": {k: v[:3] for k, v in PLACE.items()}, "HOLES": HOLES},
          open(f"{OUT}/.board_state.json","w"))
print("board written:", len(fps), "footprints", len(tracks), "segments", len(vias), "vias; W,H =", W, H)
print("pads:", {k: v for k, v in PADS.items() if k != "U1"})
