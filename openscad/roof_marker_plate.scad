// roof_marker_plate.scad
//
// Rigid backing plate for an AprilTag on the ROOF, strapped to the ~4" truss
// beam at position B so the indoor Kasa camera sees the tag when the roof is
// SHUT and loses it when the roof rolls open.
//
// Sibling of marker_plate.scad, which does the same job on the scope. Same
// reasoning, different mounting: the scope one hangs off a screwed arm aimed
// at the camera, this one straps flat to a beam and never moves.
//
// WHY THE ROOF NEEDS ONE AT ALL
// Roof-shut currently has no positive signal from this camera. The gold star
// marker appears when the roof is OPEN and is hidden behind the roof structure
// when it is shut, so "shut" would have to be inferred from that marker's
// ABSENCE — and absence is exactly what a fogged lens, a dead camera or a
// mis-pointed one also look like. A tag that is present when shut turns that
// inference into a reading. The blue closed-marker the webcam uses is not in
// this camera's field, so it cannot serve.
//
// WHY A TAG AND NOT A COLOURED SHAPE
// The camera delivers greyscale infrared on 91% of frames (943 of 1041
// sampled), and on every one of those the colour channels are identically
// zero, so anything separating by hue is blind most of the night. The scope's
// AprilTag is found on 84% of IR frames and 98% of colour ones. A tag also
// carries an ID, which is what makes it immune to the repeating truss
// structure around it: a bar can look like another bar, a tag cannot.
//
// USE 36h11 id 1. The scope is id 0, the same dictionary, so
// scripts/scope_marker_check.py reads this with no code change and compare()
// ignores ids absent from its own reference — the two references never mix.
// Generate the printable image at the exact size with:
//
//   python scripts/make_tag.py 1 --mm 76 -o tag1_76mm.png
//
// FLATNESS IS THE WHOLE JOB — the same lesson as the scope plate. A paper tag
// taped to the curved shroud read cleanly at 13:41 and was undecodable by
// 08:48 next morning: the detector still FOUND the quadrilateral but the
// decode failed, because sampling the cell grid uses a planar homography and
// the paper had buckled. The failing read was LARGER and more face-on than the
// working one, so size and angle were both ruled out. Hence a stiff plate whose
// only real job is to hold the tag in a plane.
//
// PRINT FACE DOWN. The tag face is at z=0 and comes off the build plate
// smoothest. This is not cosmetic: a ribbed or textured tag face scatters the
// IR illuminator at night, which is precisely when this reading matters. It is
// also why the pattern is a printed sticker on a smooth face rather than
// raised geometry in a second filament — relief would light unevenly under a
// raking illuminator and fatten every module with its own shadow.
//
//   openscad -o roof_marker_plate.stl roof_marker_plate.scad
//
// MOUNTING. Two zip ties through the slot pairs, around the beam. Nothing
// drilled, nothing structural, removable without trace. Fit it with the ARROW
// pointing up-slope — the tag is unambiguous to the detector on its own, but
// the arrow makes a mis-fit obvious in a photograph. Then capture a shut frame
// and record its corners the way the scope reference was recorded.

/* [Plate] */
inch          = 25.4;
plate_w       = 96;           // stays inside the ~4" (101.6 mm) beam
plate_h       = 128;          // tag plus a strap tab top and bottom
plate_thick   = 3;            // solid thickness under the tag
corner_radius = 4;

/* [Tag] */
// The printed tag INCLUDING its white quiet zone. AprilTag needs light around
// its black border or the border merges into whatever is behind the plate;
// print the image with that margin rather than trusting the beam to be pale.
tag_size      = 76;
// Shallow pocket so the sticker locates itself and its edges are protected
// from a fingernail. 0 for a plain flat face.
tag_recess    = 0.4;

/* [Stiffening] */
// On the BACK only — the face stays flat. Most of the stiffness is the rim.
rim_height    = 5;
rim_width     = 3;
rib_height    = 5;
rib_width     = 2.5;
ribs_x        = 2;
ribs_y        = 2;

/* [Straps] */
// Standard 3.6 mm zip tie plus clearance. Slots sit in the tabs, clear of the
// tag and of the stiffening rim.
tie_w         = 4.2;
tie_l         = 14;
tab_h         = 18;           // strap tab depth at each end
tie_inset     = 14;           // from each side edge

$fn = 48;
EPS = 0.01;

module rounded_plate(w, h, t, r) {
  hull() for (x = [-w/2 + r, w/2 - r], y = [-h/2 + r, h/2 - r])
    translate([x, y, 0]) cylinder(r = r, h = t);
}

module tie_slots() {
  for (sy = [-1, 1], sx = [-1, 1])
    translate([sx * (plate_w/2 - tie_inset - tie_w/2),
               sy * (plate_h/2 - tab_h/2),
               plate_thick/2])
      cube([tie_w, tie_l, plate_thick + 2], center = true);
}

// Raised on the BACK so it cannot disturb the tag face, and outside the tag
// area so it never shades it.
module up_arrow() {
  translate([0, plate_h/2 - tab_h/2, plate_thick - EPS])
    linear_extrude(1.2 + EPS)
      polygon([[0, 6], [-5, -1], [-2, -1], [-2, -6],
               [2, -6], [2, -1], [5, -1]]);
}

module stiffeners() {
  inner_w = plate_w - 2 * rim_width;
  inner_h = plate_h - 2 * tab_h;          // keep the tabs flat for the straps
  // perimeter rim, inset from the tabs
  difference() {
    translate([0, 0, plate_thick - EPS])
      rounded_plate(plate_w, inner_h, rim_height + EPS, corner_radius);
    translate([0, 0, plate_thick - 2 * EPS])
      rounded_plate(plate_w - 2 * rim_width, inner_h - 2 * rim_width,
                    rim_height + 4 * EPS, max(0.1, corner_radius - rim_width));
  }
  // Ribs sit ON the plate (z from plate_thick up), so they are centred in X
  // and Y only -- a cube centred in Z as well would sink half its height into
  // the tag face, which is the one surface that must stay untouched.
  for (i = [1 : ribs_x])
    translate([0, -inner_h/2 + i * inner_h / (ribs_x + 1), plate_thick - EPS])
      linear_extrude(rib_height + EPS)
        square([inner_w, rib_width], center = true);
  for (i = [1 : ribs_y])
    translate([-plate_w/2 + i * plate_w / (ribs_y + 1), 0, plate_thick - EPS])
      linear_extrude(rib_height + EPS)
        square([rib_width, inner_h], center = true);
}

difference() {
  union() {
    rounded_plate(plate_w, plate_h, plate_thick, corner_radius);
    stiffeners();
    up_arrow();
  }
  // tag pocket, in the FACE (z=0, the build-plate side)
  if (tag_recess > 0)
    translate([0, 0, -EPS])
      linear_extrude(tag_recess + EPS)
        square([tag_size, tag_size], center = true);
  tie_slots();
}

echo(str("plate ", plate_w, " x ", plate_h, " x ", plate_thick, " mm"));
echo(str("tag area ", tag_size, " mm sq, recess ", tag_recess, " mm"));
