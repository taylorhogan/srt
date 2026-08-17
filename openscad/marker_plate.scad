// marker_plate.scad
//
// Rigid backing plate for the ArUco/AprilTag scope marker, with an arm that
// carries a single mounting screw.
//
// Why this exists: a paper tag taped to the curved shroud read cleanly for
// about 19 hours and then stopped decoding. The detector still FOUND the
// marker's quadrilateral but could not read its bits, because sampling the
// cell grid uses a planar homography and the buckled paper was no longer
// planar. Size and viewing angle were both ruled out — the failing read was
// larger and closer to face-on than the working one. Flatness was the whole
// problem, so the plate's only real job is to stay flat and stand proud of the
// shroud's curve rather than conforming to it.
//
// Print face down. The tag goes on the underside (z=0), which comes off the
// build plate smoothest — a textured or ribbed tag face would scatter the IR
// illuminator at night and defeat the point.
//
//   openscad -o marker_plate.stl marker_plate.scad
//   openscad -D plate_size=127 -o marker_plate_5in.stl marker_plate.scad

/* [Plate] */
inch            = 25.4;
plate_size      = 4 * inch;   // 4" square, matching the printed tag
plate_thick     = 3;          // solid thickness under the tag
corner_radius   = 4;          // rounds the corners; 0 for square

/* [Tag recess] */
// Optional shallow pocket on the tag face so the print locates the tag and
// protects its edges. Leave at 0 for a flat face and glue the tag on.
// Only useful if plate_size is LARGER than tag_size — with both at 4" the
// pocket spans the whole face and locates nothing.
tag_recess      = 0;          // depth, mm
tag_size        = 4 * inch;   // tag outer size including its white quiet zone

/* [Stiffening] */
rim_height      = 5;          // perimeter lip on the back — most of the stiffness
rim_width       = 3;
rib_height      = 5;          // internal ribs
rib_width       = 2.5;
ribs_x          = 3;          // ribs running along X
ribs_y          = 3;          // ribs running along Y
diagonal_ribs   = true;       // adds torsional stiffness the grid alone lacks

/* [Arm] */
arm_height      = 2 * inch;   // above the plate's back face
arm_width       = 25;
arm_thick       = 6;
arm_edge        = "y";        // which edge the arm sits on: "y" or "x"

/* [Screw] */
hole_dia        = 5.4;        // M5 clearance; 4.4 for M4, 6.4 for M6
hole_from_top   = 12;         // centre of the hole below the arm's tip
counterbore_dia = 0;          // 0 = plain through hole
counterbore_dep = 0;

/* [Gussets] */
// The arm root is the only place this part can plausibly break: a long lever
// meeting a thin plate. Two triangular webs carry that moment into the plate.
gusset_run      = 28;         // how far the web reaches across the plate
gusset_rise     = 34;         // how far it climbs the arm
gusset_thick    = 3;

/* [Quality] */
$fn             = 64;

// ---------------------------------------------------------------- geometry

module rounded_square(size, r, h) {
    if (r <= 0) {
        translate([-size/2, -size/2, 0]) cube([size, size, h]);
    } else {
        hull() for (sx = [-1, 1], sy = [-1, 1])
            translate([sx * (size/2 - r), sy * (size/2 - r), 0])
                cylinder(r = r, h = h);
    }
}

module plate() {
    difference() {
        rounded_square(plate_size, corner_radius, plate_thick);
        if (tag_recess > 0)
            translate([0, 0, -0.01])
                rounded_square(tag_size, max(0, corner_radius - 1), tag_recess + 0.01);
    }
}

module rim() {
    difference() {
        translate([0, 0, plate_thick])
            rounded_square(plate_size, corner_radius, rim_height);
        translate([0, 0, plate_thick - 0.01])
            rounded_square(plate_size - 2 * rim_width,
                           max(0, corner_radius - rim_width), rim_height + 0.02);
    }
}

module ribs() {
    span = plate_size - 2 * rim_width;
    // Ribs run between the rims rather than to the plate edge, so they tie the
    // rim together instead of just adding material near a free edge.
    for (i = [1 : ribs_x])
        translate([-span/2 + i * span / (ribs_x + 1) - rib_width/2,
                   -span/2, plate_thick])
            cube([rib_width, span, rib_height]);
    for (i = [1 : ribs_y])
        translate([-span/2, -span/2 + i * span / (ribs_y + 1) - rib_width/2,
                   plate_thick])
            cube([span, rib_width, rib_height]);
    if (diagonal_ribs)
        // A rectangular grid resists bending but racks in torsion; the
        // diagonals are what stop the plate twisting about the arm.
        intersection() {
            translate([0, 0, plate_thick])
                rounded_square(plate_size - 2 * rim_width,
                               max(0, corner_radius - rim_width), rib_height);
            // The union() is load-bearing: intersection() intersects ALL its
            // children, so a bare for-loop here would give the two diagonals
            // intersected with each other — a small lozenge at the centre
            // instead of a cross.
            union()
                for (a = [45, -45])
                    rotate([0, 0, a])
                        translate([-plate_size, -rib_width/2, 0])
                            cube([2 * plate_size, rib_width,
                                  plate_thick + rib_height]);
        }
}

module arm_body() {
    base = plate_thick;                    // arm starts at the plate's back face
    difference() {
        translate([-arm_width/2, plate_size/2 - arm_thick, 0])
            cube([arm_width, arm_thick, base + arm_height]);
        // screw hole, through the arm's thickness
        translate([0, plate_size/2 + 0.01, base + arm_height - hole_from_top])
            rotate([90, 0, 0])
                cylinder(d = hole_dia, h = arm_thick + 0.02);
        if (counterbore_dia > 0)
            translate([0, plate_size/2 + 0.01, base + arm_height - hole_from_top])
                rotate([90, 0, 0])
                    cylinder(d = counterbore_dia, h = counterbore_dep + 0.01);
    }
}

module gussets() {
    base = plate_thick;
    for (sx = [-1, 1])
        translate([sx * (arm_width/2 - gusset_thick/2) - gusset_thick/2, 0, 0])
            rotate([90, 0, 90])
                linear_extrude(height = gusset_thick)
                    polygon([[plate_size/2 - arm_thick,      base],
                             [plate_size/2 - arm_thick,      base + gusset_rise],
                             [plate_size/2 - arm_thick - gusset_run, base]]);
}

module marker_plate() {
    union() {
        plate();
        rim();
        ribs();
        arm_body();
        gussets();
    }
}

// The arm is described as sitting on the +Y edge; rotating the whole part is
// simpler and less error-prone than parameterising every reference to it.
if (arm_edge == "x") rotate([0, 0, 90]) marker_plate();
else                 marker_plate();

echo(str("plate ", plate_size, " mm (", plate_size/inch, " in) square, ",
         plate_thick, " mm thick"));
echo(str("arm ", arm_height, " mm (", arm_height/inch, " in) above the back face, ",
         "hole ", hole_dia, " mm, ", hole_from_top, " mm below the tip"));
echo(str("overall height ", plate_thick + arm_height, " mm"));
