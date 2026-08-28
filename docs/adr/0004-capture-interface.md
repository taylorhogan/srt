# ADR 0004: Capture behind an interface; NINA first

Status: accepted 2026-08-28

A CaptureSession protocol owns cool/filter/expose/focus/dither; NINA is
its first implementation and its autofocus stays NINA's for the
foreseeable future (HocusFocus is genuinely good and rewrites of focus
loops disappoint). A native QHY-SDK session can replace it later without
touching the machine. The seam exists so the choice never has to be made
under pressure.
