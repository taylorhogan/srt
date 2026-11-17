"""Sky IR sensor carrier board: KiCad 8 project generator (part A: common + schematic)."""
import uuid, json, os, sys

OUT = sys.argv[1]
ROW = 22.86          # DevKitC-1 header row spacing (0.9 in). CONFIRM on the board before ordering.
DATE = "2026-11-17"
def U(): return str(uuid.uuid4())
ROOT = U()
F = "(font (size 1.27 1.27))"

# ---------------- netlist (the one truth for both files) ----------------
J1_NAMES = ["3V3","3V3","RST","4","5","6","7","15","16","17","18","8","3","46","9","10","11","12","13","14","5V","GND"]
J3_NAMES = ["GND","TX","RX","1","2","42","41","40","39","38","37","36","35","0","45","48","47","21","20","19","GND","GND"]
NETS = ["GND","+3V3","+5V","SDA","SCL","GPIO10","GATE","DRAIN"]
# component -> {pin: net}
CONN = {
 "U1": {"1":"+3V3","2":"+3V3","12":"SDA","15":"SCL","16":"GPIO10","21":"+5V","22":"GND","23":"GND","43":"GND","44":"GND"},
 "J2": {"1":"+3V3","2":"GND","3":"SCL","4":"SDA"},   # MLX90614 leads
 "J4": {"1":"+3V3","2":"GND","3":"SCL","4":"SDA"},   # BME280 breakout VIN GND SCL SDA
 "J5": {"1":"+5V","2":"DRAIN"},                      # heater
 "R1": {"1":"+3V3","2":"SDA"}, "R2": {"1":"+3V3","2":"SCL"},
 "R3": {"1":"GPIO10","2":"GATE"}, "R4": {"1":"GATE","2":"GND"},
 "Q1": {"1":"GATE","2":"DRAIN","3":"GND"},           # IRLZ44N G D S
}
VALUES = {"U1":"ESP32-S3-DevKitC-1","J2":"MLX90614","J4":"BME280","J5":"Heater 47R","R1":"4.7k","R2":"4.7k","R3":"220","R4":"10k","Q1":"IRLZ44N"}
FOOTS = {"U1":"sky_ir:ESP32-S3-DevKitC-1_Sockets","J2":"sky_ir:PinHeader_1x04","J4":"sky_ir:PinHeader_1x04","J5":"sky_ir:PinHeader_1x02",
         "R1":"sky_ir:R_Axial_P10.16mm","R2":"sky_ir:R_Axial_P10.16mm","R3":"sky_ir:R_Axial_P10.16mm","R4":"sky_ir:R_Axial_P10.16mm","Q1":"sky_ir:TO-220-3_Vertical"}
SYMU = {r: U() for r in CONN}     # schematic symbol uuids, reused as footprint paths

# ---------------- symbol library ----------------
def pin(kind, x, y, ang, L, name, num, hide_name=False):
    hn = " hide" if hide_name else ""
    return f'(pin {kind} line (at {x} {y} {ang}) (length {L}) (name "{name}" (effects {F}{hn})) (number "{num}" (effects {F})))'
def props(ref, val, yr, yv):
    return (f'(property "Reference" "{ref}" (at 0 {yr} 0) (effects {F}))\n  (property "Value" "{val}" (at 0 {yv} 0) (effects {F}))\n'
            f'  (property "Footprint" "" (at 0 0 0) (effects {F} hide))\n  (property "Datasheet" "" (at 0 0 0) (effects {F} hide))\n'
            f'  (property "Description" "" (at 0 0 0) (effects {F} hide))')
STK = '(stroke (width 0.254) (type default)) (fill (type none))'
def sym_R():
    return (f'(symbol "sky_ir:R" (pin_numbers hide) (pin_names (offset 0) hide) (exclude_from_sim no) (in_bom yes) (on_board yes)\n  {props("R","R",2.54,-2.54)}\n'
            f'  (symbol "R_0_1" (rectangle (start -2.54 -1.016) (end 2.54 1.016) {STK}))\n'
            f'  (symbol "R_1_1" {pin("passive",-3.81,0,0,1.27,"~","1")} {pin("passive",3.81,0,180,1.27,"~","2")}))')
def sym_conn(name, names):
    n = len(names); ys = [ (n-1)*1.27 - i*2.54 for i in range(n)]
    pins = " ".join(pin("passive",-5.08,y,0,3.81,nm,str(i+1)) for i,(y,nm) in enumerate(zip(ys,names)))
    h = n*1.27
    return (f'(symbol "sky_ir:{name}" (pin_names (offset 1.016)) (exclude_from_sim no) (in_bom yes) (on_board yes)\n  {props("J",name,h+1.27,-h-1.27)}\n'
            f'  (symbol "{name}_0_1" (rectangle (start -1.27 {h}) (end 1.27 {-h}) {STK}))\n  (symbol "{name}_1_1" {pins}))')
def sym_esp():
    L = [pin("passive",-12.7, 26.67-i*2.54, 0, 2.54, J1_NAMES[i], str(i+1)) for i in range(22)]
    R = [pin("passive", 12.7, 26.67-i*2.54, 180, 2.54, J3_NAMES[i], str(i+23)) for i in range(22)]
    return (f'(symbol "sky_ir:ESP32-S3-DevKitC-1" (pin_names (offset 1.016)) (exclude_from_sim no) (in_bom yes) (on_board yes)\n  {props("U","ESP32-S3-DevKitC-1",31.75,-31.75)}\n'
            f'  (symbol "ESP32-S3-DevKitC-1_0_1" (rectangle (start -10.16 29.21) (end 10.16 -29.21) {STK}))\n'
            f'  (symbol "ESP32-S3-DevKitC-1_1_1" {" ".join(L+R)}))')
def sym_nmos():
    body = (f'(polyline (pts (xy -2.54 -2.54) (xy -2.54 2.54)) {STK}) (polyline (pts (xy -1.27 -2.54) (xy -1.27 2.54)) {STK}) '
            f'(polyline (pts (xy 0 2.54) (xy 0 1.905) (xy -1.27 1.905)) {STK}) (polyline (pts (xy 0 -2.54) (xy 0 -1.905) (xy -1.27 -1.905)) {STK}) '
            f'(polyline (pts (xy -1.27 0) (xy 0.635 0) (xy 0.635 -1.905)) {STK}) (circle (center -0.3 0) (radius 3.3) {STK})')
    return (f'(symbol "sky_ir:NMOS_GDS" (pin_names (offset 1.016) hide) (exclude_from_sim no) (in_bom yes) (on_board yes)\n  {props("Q","NMOS",5.08,-6.35)}\n'
            f'  (symbol "NMOS_GDS_0_1" {body})\n'
            f'  (symbol "NMOS_GDS_1_1" {pin("input",-5.08,0,0,2.54,"G","1")} {pin("passive",0,5.08,270,2.54,"D","2")} {pin("passive",0,-5.08,90,2.54,"S","3")}))')
LIBSYMS = [sym_R(), sym_conn("MLX90614", ["VDD","VSS","SCL","SDA"]), sym_conn("BME280", ["VIN","GND","SCL","SDA"]),
           sym_conn("Heater", ["+5V","DRAIN"]), sym_esp(), sym_nmos()]
LIBID = {"U1":"sky_ir:ESP32-S3-DevKitC-1","J2":"sky_ir:MLX90614","J4":"sky_ir:BME280","J5":"sky_ir:Heater","R1":"sky_ir:R","R2":"sky_ir:R","R3":"sky_ir:R","R4":"sky_ir:R","Q1":"sky_ir:NMOS_GDS"}

# ---------------- schematic placement ----------------
items = []
def wire(a, b): items.append(f'(wire (pts (xy {a[0]} {a[1]}) (xy {b[0]} {b[1]})) (stroke (width 0) (type default)) (uuid "{U()}"))')
def label(txt, p, ang):
    just = {0:"left bottom",180:"right bottom",90:"left bottom",270:"right bottom"}[ang]
    items.append(f'(label "{txt}" (at {p[0]} {p[1]} {ang}) (effects {F} (justify {just})) (uuid "{U()}"))')
def nc(p): items.append(f'(no_connect (at {p[0]} {p[1]}) (uuid "{U()}"))')
def place(ref, x, y, pins, refpos, valpos):
    pp = "\n  ".join(f'(pin "{n}" (uuid "{U()}"))' for n in pins)
    items.append(f'''(symbol (lib_id "{LIBID[ref]}") (at {x} {y} 0) (unit 1) (exclude_from_sim no) (in_bom yes) (on_board yes) (dnp no)
  (uuid "{SYMU[ref]}")
  (property "Reference" "{ref}" (at {refpos[0]} {refpos[1]} 0) (effects {F}))
  (property "Value" "{VALUES[ref]}" (at {valpos[0]} {valpos[1]} 0) (effects {F}))
  (property "Footprint" "{FOOTS[ref]}" (at {x} {y} 0) (effects {F} hide))
  (property "Datasheet" "" (at {x} {y} 0) (effects {F} hide))
  (property "Description" "" (at {x} {y} 0) (effects {F} hide))
  {pp}
  (instances (project "sky_ir" (path "/{ROOT}" (reference "{ref}") (unit 1)))))''')
SCH_POS = {}   # for the plot
# U1
ux, uy = 81.28, 88.9
place("U1", ux, uy, [str(i) for i in range(1,45)], (ux, uy-33.02), (ux, uy+33.02))
for i in range(22):
    y = round(uy - 26.67 + i*2.54, 2)
    lp = (round(ux-12.7,2), y); rp = (round(ux+12.7,2), y)
    n = CONN["U1"].get(str(i+1))
    if n: wire(lp, (lp[0]-7.62, y)); label(n, (lp[0]-7.62, y), 180)
    else: nc(lp)
    n = CONN["U1"].get(str(i+23))
    if n: wire(rp, (rp[0]+7.62, y)); label(n, (rp[0]+7.62, y), 0)
    else: nc(rp)
# resistors
for ref, y in [("R1",66.04),("R2",73.66),("R3",101.6),("R4",109.22)]:
    x = 129.54
    place(ref, x, y, ["1","2"], (x, y-2.54), (x, y+2.54))
    a = (x-3.81, y); b = (x+3.81, y)
    wire(a, (a[0]-5.08, y)); label(CONN[ref]["1"], (a[0]-5.08, y), 180)
    wire(b, (b[0]+5.08, y)); label(CONN[ref]["2"], (b[0]+5.08, y), 0)
# Q1
qx, qy = 154.94, 93.98
place("Q1", qx, qy, ["1","2","3"], (qx+5.08, qy-2.54), (qx+5.08, qy+2.54))
wire((qx-5.08, qy), (qx-10.16, qy)); label("GATE", (qx-10.16, qy), 180)
wire((qx, qy-5.08), (qx, qy-10.16)); label("DRAIN", (qx, qy-10.16), 90)
wire((qx, qy+5.08), (qx, qy+10.16)); label("GND", (qx, qy+10.16), 270)
# connectors
for ref, cy in [("J2",66.04),("J4",86.36),("J5",127.0)]:
    cx = 190.5; n = len(CONN[ref])
    place(ref, cx, cy, [str(i+1) for i in range(n)], (cx, cy-n*1.27-2.54), (cx, cy+n*1.27+2.54))
    for i in range(n):
        y = round(cy - (n-1)*1.27 + i*2.54, 2); p = (cx-5.08, y)
        wire(p, (p[0]-7.62, y)); label(CONN[ref][str(i+1)], (p[0]-7.62, y), 180)
items.append(f'(text "Sky IR sensor carrier: ESP32-S3-DevKitC-1 in two 1x22 sockets, I2C on GPIO8/9 with 4.7k pull-ups,\\nGPIO10 drives an IRLZ44N that switches the 47 ohm window heater from the 5 V pin. MLX90614, BME280 and heater on headers (off-board leads).\\nUnused DevKit pins are left open on purpose." (exclude_from_sim no) (at 60.96 160.02 0) (effects {F} (justify left bottom)) (uuid "{U()}"))')

sch = f'''(kicad_sch (version 20231120) (generator "eeschema") (generator_version "8.0")
  (uuid "{ROOT}")
  (paper "A4")
  (title_block (title "Sky IR sensor carrier") (date "{DATE}") (rev "1") (company "Iris observatory") (comment 1 "Row spacing of the DevKit sockets is {ROW} mm: confirm on the board before ordering"))
  (lib_symbols
  {chr(10).join(LIBSYMS)}
  )
  {chr(10).join(items)}
  (sheet_instances (path "/" (page "1")))
)
'''
os.makedirs(OUT, exist_ok=True)
open(f"{OUT}/sky_ir.kicad_sch","w").write(sch)
# standalone symbol library too
open(f"{OUT}/sky_ir.kicad_sym","w").write('(kicad_symbol_lib (version 20231120) (generator "kicad_symbol_editor") (generator_version "8.0")\n' + "\n".join(s.replace('"sky_ir:', '"', 1) for s in LIBSYMS) + "\n)\n")
open(f"{OUT}/sym-lib-table","w").write('(sym_lib_table (version 7)\n  (lib (name "sky_ir")(type "KiCad")(uri "${KIPRJMOD}/sky_ir.kicad_sym")(options "")(descr "Sky IR sensor symbols"))\n)\n')
json.dump({"ROW":ROW,"ROOT":ROOT,"SYMU":SYMU,"CONN":CONN,"NETS":NETS,"VALUES":VALUES,"FOOTS":FOOTS,"DATE":DATE}, open(f"{OUT}/.gen_state.json","w"))
print("schematic written", len(items), "items")
