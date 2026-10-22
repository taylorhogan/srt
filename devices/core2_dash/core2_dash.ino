// Iris desk display: M5Stack Core2 v1.3 (2026-10-22).
//
// Polls GET /api/dash on the chat server every 30 s and paints it. Display
// only: touch and the buttons do nothing, and the payload carries no command.
// The colors are decided by the server (cmd_processing/dash.py): g fine,
// a look at it, r act now, x no data. If no good reply has arrived for
// STALE_MS the whole screen goes grey with the age, so an old "SHUT" is never
// read as current.
//
// Build: Arduino IDE or arduino-cli, board "M5Core2" (M5Stack board package),
// libraries M5Unified and ArduinoJson 7. Copy secrets.h.example to secrets.h.
// M5Unified, not the old M5Core2 library: v1.3 has the AXP2101 power chip.

#include <M5Unified.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include "secrets.h"   // WIFI_SSID, WIFI_PASS, DASH_URL

static const uint32_t POLL_MS = 30000;
static const uint32_t STALE_MS = 180000;
static const uint8_t BRIGHTNESS = 90;          // 0-255

static JsonDocument doc;
static bool haveData = false;
static uint32_t lastOk = 0, lastPoll = 0;
static String lastErr = "starting";

static uint16_t colorFor(const char* c) {
  if (!c) return TFT_DARKGREY;
  switch (c[0]) {
    case 'g': return TFT_GREEN;
    case 'a': return TFT_ORANGE;
    case 'r': return TFT_RED;
    default:  return TFT_DARKGREY;
  }
}

static void fetch() {
  if (WiFi.status() != WL_CONNECTED) { lastErr = "no wifi"; return; }
  HTTPClient http;
  http.setTimeout(15000);                      // first call after idle can take ~5 s
  if (!http.begin(DASH_URL)) { lastErr = "bad url"; return; }
  int code = http.GET();
  if (code == 200) {
    JsonDocument fresh;
    DeserializationError e = deserializeJson(fresh, http.getStream());
    if (!e && fresh["ok"].as<bool>()) {
      doc = fresh; haveData = true; lastOk = millis(); lastErr = "";
    } else {
      lastErr = e ? "bad json" : "server not ok";
    }
  } else {
    lastErr = "http " + String(code);
  }
  http.end();
}

// One labelled row: small grey label, big value in its color. The built-in
// font is 6 px wide per text size, so a value longer than 13 characters drops
// to size 2 (20 characters) rather than running off the right edge.
static void row(int y, const char* label, const String& value, uint16_t color) {
  auto& d = M5.Display;
  d.setTextColor(TFT_LIGHTGREY, TFT_BLACK);
  d.setTextSize(1);
  d.setCursor(8, y + 8);
  d.print(label);
  d.setTextColor(color, TFT_BLACK);
  bool big = value.length() * 18 <= 320 - 78 - 4;
  d.setTextSize(big ? 3 : 2);
  d.setCursor(78, big ? y : y + 4);
  d.print(value.substring(0, 20));
}

static void paint() {
  auto& d = M5.Display;
  d.startWrite();
  d.fillScreen(TFT_BLACK);
  bool stale = !haveData || millis() - lastOk > STALE_MS;

  // Top band: overall level, clock, night state.
  uint16_t band = stale ? TFT_DARKGREY : colorFor(doc["level"]);
  d.fillRect(0, 0, 320, 30, band);
  d.setTextColor(TFT_BLACK, band);
  d.setTextSize(2);
  d.setCursor(8, 8);
  d.print("IRIS");
  if (haveData) {
    d.setCursor(70, 8);
    d.print(doc["t"].as<const char*>());
    String night = doc["night"]["v"] | "";
    d.setCursor(320 - 12 * night.length() - 8, 8);
    d.print(night);
  }

  if (!haveData) {
    row(60, "STATUS", "NO DATA", TFT_DARKGREY);
    d.setTextSize(2); d.setTextColor(TFT_LIGHTGREY, TFT_BLACK);
    d.setCursor(8, 120); d.print(lastErr);
    d.endWrite();
    return;
  }

  auto grey = [&](uint16_t c) { return stale ? (uint16_t)TFT_DARKGREY : c; };

  // Roof (north camera's newest read) with its age.
  String roof = doc["roof"]["v"] | "?";
  if (!doc["roof"]["age"].isNull()) {
    int age = doc["roof"]["age"];
    roof += age < 60 ? "  " + String(age) + "m" : "  " + String(age / 60) + "h";
  }
  row(40, "ROOF", roof, grey(colorFor(doc["roof"]["c"])));
  row(78, "SCOPE", doc["scope"]["v"] | "?", grey(colorFor(doc["scope"]["c"])));

  // Now imaging, or what comes next.
  if (!doc["now"].isNull()) {
    String now = String((const char*)doc["now"]["dso"]) + " " + (const char*)doc["now"]["filter"]
                 + " " + String((int)doc["now"]["n"]);
    if (!doc["now"]["of"].isNull()) now += "/" + String((int)doc["now"]["of"]);
    row(116, "NOW", now, grey(TFT_CYAN));
  } else if (!doc["next"].isNull()) {
    row(116, "NEXT", String((const char*)doc["next"]["dso"]) + " " + (const char*)doc["next"]["at"],
        grey(TFT_WHITE));
  } else {
    row(116, "NEXT", doc["tonight"]["image"] ? "-" : "no imaging", grey(TFT_WHITE));
  }

  // Roof window and disk on one small line.
  d.setTextSize(2);
  d.setTextColor(grey(TFT_LIGHTGREY), TFT_BLACK);
  d.setCursor(8, 162);
  const char* op = doc["tonight"]["opens"] | "--:--";
  const char* cl = doc["tonight"]["closes"] | "--:--";
  d.printf("open %s  close %s", op, cl);
  if (!doc["disk"].isNull()) {
    d.setTextColor(grey(colorFor(doc["disk"]["c"])), TFT_BLACK);
    d.setCursor(8, 186);
    d.printf("disk %d GB", (int)doc["disk"]["gb"]);
  }

  // Bottom line: alerts, or how old the data is when it has gone stale.
  d.setCursor(8, 214);
  if (stale) {
    d.setTextColor(TFT_ORANGE, TFT_BLACK);
    d.printf("NO UPDATE %lus  %s", (unsigned long)((millis() - lastOk) / 1000), lastErr.c_str());
  } else if (doc["alerts"].size() > 0) {
    d.setTextColor(colorFor(doc["level"]), TFT_BLACK);
    d.print(doc["alerts"][0].as<const char*>());
  }
  d.endWrite();
}

void setup() {
  auto cfg = M5.config();
  M5.begin(cfg);
  M5.Display.setRotation(1);
  M5.Display.setBrightness(BRIGHTNESS);
  WiFi.mode(WIFI_STA);
  WiFi.setAutoReconnect(true);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  paint();
}

void loop() {
  M5.update();                                 // battery/power housekeeping only
  if (lastPoll == 0 || millis() - lastPoll >= POLL_MS) {
    lastPoll = millis();
    fetch();
    paint();
  }
  delay(200);
}
