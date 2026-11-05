// The north camera's roof AprilTag (36h11, id 2) as a 3D print: the BLACK
// cells only, one connected piece, to lay on a white background (2026-11-05).
// Sibling of scope_tag_black_id0.scad; the backing is
// north_roof_tag_white_back.scad.
//
// Same size as the paper tag on the moving south truss (docs/ROOF_STATE_SENSING.md):
// 150 mm including its one module of white quiet zone, so 10 modules of 15 mm,
// and this part -- the 8 x 8 black region -- is 120 mm square. The white
// quiet zone is the background you lay it on: leave at least one module
// (15 mm) of white showing on every side, or the detector cannot find the
// tag's outline.
//
// One piece with no patches: every black cell of id 2 shares an edge with
// the border ring or another black cell (checked 4-connected), so
// corner_patches is empty. Verified by rendering this geometry and decoding
// it as id 2 (scripts/scope_tag_scad_check.py openscad/north_roof_tag_black_id2.scad).
//
// Printing (as for the scope tag):
//   * Black filament that is black in the NEAR INFRARED. Iris North reads
//     this tag in pure IR most of the night, and some black dyes are close to
//     transparent there. Check a test square under the camera at night first.
//   * Matte face: print the visible face down on a textured sheet, or on the
//     bare bed and scuff it lightly.
//   * The pattern is not mirror-symmetric; mirror_for_face_down handles a
//     face-down print.

tag_size  = 150;              // whole tag including the white quiet zone
modules   = 10;               // 8 tag modules + 1 quiet module each side
module_mm = tag_size / modules;   // 15
thickness = 1.6;              // mm
patch     = 1.6;              // corner patch, mm (keep under 2 x 1.32)
mirror_for_face_down = true;  // true when the visible face prints on the bed
eps = 0.01;                   // overlap so touching cells fuse into one solid

// 1 = black, row 0 at the TOP as the tag is seen; tag 36h11 id 2.
bits = [
  [1,1,1,1,1,1,1,1],
  [1,1,0,0,0,1,1,1],
  [1,1,1,0,1,1,1,1],
  [1,0,1,1,0,1,1,1],
  [1,1,1,1,1,1,0,1],
  [1,1,0,1,1,0,1,1],
  [1,0,0,0,1,0,0,1],
  [1,1,1,1,1,1,1,1],
];

// Lattice corners [row, col] (row 0 = top edge) that get a patch: none for id 2.
corner_patches = [];

N = len(bits);

module cell(r, c) {
  // x right, y up; row 0 at the top
  translate([c * module_mm - eps, (N - 1 - r) * module_mm - eps, 0])
    cube([module_mm + 2 * eps, module_mm + 2 * eps, thickness]);
}

module corner_patch(r, c) {
  translate([c * module_mm - patch / 2, (N - r) * module_mm - patch / 2, 0])
    cube([patch, patch, thickness]);
}

module tag_black() {
  union() {
    for (r = [0 : N - 1], c = [0 : N - 1])
      if (bits[r][c] == 1) cell(r, c);
    for (p = corner_patches) corner_patch(p[0], p[1]);
  }
}

if (mirror_for_face_down)
  translate([N * module_mm, 0, 0]) mirror([1, 0, 0]) tag_black();
else
  tag_black();
