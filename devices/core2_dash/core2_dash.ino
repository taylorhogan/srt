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
// Portrait: 0 or 2. If the picture is upside down for the way the unit
// stands (USB-C at the bottom), switch to the other one.
static const uint8_t ROTATION = 2;

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

// Portrait (2026-10-24): the Core2 stands on its side so the USB-C cable
// leaves from the bottom. 240 x 320, so each row is a small grey label with
// its value BELOW it, full width. The built-in font is 6 px wide per text
// size: a value of up to 13 characters is size 3, longer drops to size 2
// (20 characters) rather than running off the right edge.
static const int W = 240;

static void row(int y, const char* label, const String& value, uint16_t color) {
  auto& d = M5.Display;
  d.setTextColor(TFT_LIGHTGREY, TFT_BLACK);
  d.setTextSize(1);
  d.setCursor(8, y);
  d.print(label);
  d.setTextColor(color, TFT_BLACK);
  bool big = value.length() * 18 <= W - 12;
  d.setTextSize(big ? 3 : 2);
  d.setCursor(8, big ? y + 11 : y + 14);
  d.print(value.substring(0, 20));
}

static void paint() {
  auto& d = M5.Display;
  d.startWrite();
  d.fillScreen(TFT_BLACK);
  bool stale = !haveData || millis() - lastOk > STALE_MS;

  // Top band: overall level and the clock; the night state on the line below.
  uint16_t band = stale ? TFT_DARKGREY : colorFor(doc["level"]);
  d.fillRect(0, 0, W, 30, band);
  d.setTextColor(TFT_BLACK, band);
  d.setTextSize(2);
  d.setCursor(8, 8);
  d.print("IRIS");
  if (haveData) {
    d.setCursor(W - 12 * 5 - 8, 8);
    d.print(doc["t"].as<const char*>());
  }

  if (!haveData) {
    row(60, "STATUS", "NO DATA", TFT_DARKGREY);
    d.setTextSize(2); d.setTextColor(TFT_LIGHTGREY, TFT_BLACK);
    d.setCursor(8, 110); d.print(lastErr);
    // What the router needs for a DHCP reservation, and what the chat's
    // allowlist will be given: shown until the first good reply.
    d.setTextSize(1);
    d.setCursor(8, 150); d.print("IP  "); d.print(WiFi.localIP().toString());
    d.setCursor(8, 166); d.print("MAC "); d.print(WiFi.macAddress());
    d.setCursor(8, 182); d.print("URL "); d.print(DASH_URL);
    d.endWrite();
    return;
  }

  auto grey = [&](uint16_t c) { return stale ? (uint16_t)TFT_DARKGREY : c; };

  d.setTextSize(2);
  d.setTextColor(grey(colorFor(doc["night"]["c"])), TFT_BLACK);
  d.setCursor(8, 38);
  d.print(doc["night"]["v"] | "");

  // Roof (north camera's newest read) with its age.
  String roof = doc["roof"]["v"] | "?";
  if (!doc["roof"]["age"].isNull()) {
    int age = doc["roof"]["age"];
    roof += age < 60 ? "  " + String(age) + "m" : "  " + String(age / 60) + "h";
  }
  row(64, "ROOF", roof, grey(colorFor(doc["roof"]["c"])));
  row(108, "SCOPE", doc["scope"]["v"] | "?", grey(colorFor(doc["scope"]["c"])));

  // Now imaging, or what comes next.
  if (!doc["now"].isNull()) {
    String now = String((const char*)doc["now"]["dso"]) + " " + (const char*)doc["now"]["filter"]
                 + " " + String((int)doc["now"]["n"]);
    if (!doc["now"]["of"].isNull()) now += "/" + String((int)doc["now"]["of"]);
    row(152, "NOW", now, grey(TFT_CYAN));
  } else if (!doc["next"].isNull()) {
    row(152, "NEXT", String((const char*)doc["next"]["dso"]) + " " + (const char*)doc["next"]["at"],
        grey(TFT_WHITE));
  } else {
    row(152, "NEXT", doc["tonight"]["image"] ? "-" : "no imaging", grey(TFT_WHITE));
  }

  // Roof window and disk, one per line.
  d.setTextSize(2);
  d.setTextColor(grey(TFT_LIGHTGREY), TFT_BLACK);
  const char* op = doc["tonight"]["opens"] | "--:--";
  const char* cl = doc["tonight"]["closes"] | "--:--";
  d.setCursor(8, 202); d.printf("open  %s", op);
  d.setCursor(8, 224); d.printf("close %s", cl);
  if (!doc["disk"].isNull()) {
    d.setTextColor(grey(colorFor(doc["disk"]["c"])), TFT_BLACK);
    d.setCursor(8, 246);
    d.printf("disk  %d GB", (int)doc["disk"]["gb"]);
  }

  // Bottom: alerts, or how old the data is when it has gone stale.
  if (stale) {
    d.setTextColor(TFT_ORANGE, TFT_BLACK);
    d.setCursor(8, 282);
    d.printf("NO UPDATE %lus", (unsigned long)((millis() - lastOk) / 1000));
    d.setTextSize(1);
    d.setCursor(8, 304);
    d.print(lastErr);
  } else if (doc["alerts"].size() > 0) {
    d.setTextColor(colorFor(doc["level"]), TFT_BLACK);
    d.setCursor(8, 290);
    d.print(String(doc["alerts"][0].as<const char*>()).substring(0, 19));
  }
  d.endWrite();
}

void setup() {
  auto cfg = M5.config();
  M5.begin(cfg);
  M5.Display.setRotation(ROTATION);
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
