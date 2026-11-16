/*
 * ESP32 MQTT publisher for Design Project AIoT loop (secondary validation).
 *
 * Publish cadence: ~1 Hz JSON telemetry to topic jitai/device/{DEVICE_ID}/telemetry
 * Subscribe (optional): jitai/device/{DEVICE_ID}/recommendation
 *
 * Flash with Arduino IDE / PlatformIO (board: ESP32 Dev Module).
 * Copy secrets.h.example -> secrets.h and fill WiFi + MQTT broker.
 *
 * Science claim is unchanged: ranking evaluation remains simulation-first.
 */

#include <WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>

#include "secrets.h"

#ifndef DEVICE_ID
#define DEVICE_ID "esp32-01"
#endif

#ifndef MQTT_PORT
#define MQTT_PORT 1883
#endif

#ifndef TELEMETRY_HZ
#define TELEMETRY_HZ 1
#endif

WiFiClient wifiClient;
PubSubClient mqtt(wifiClient);

char topicTelemetry[96];
char topicReco[96];
char topicStatus[96];

unsigned long lastPublishMs = 0;
unsigned long bootMs = 0;
uint32_t seq = 0;

// Soft "sensor" proxies (replace with real IMU / PPG drivers as needed).
float fakeSteps = 0.0f;
float fakeHr = 72.0f;

void setTopics() {
  snprintf(topicTelemetry, sizeof(topicTelemetry), "jitai/device/%s/telemetry", DEVICE_ID);
  snprintf(topicReco, sizeof(topicReco), "jitai/device/%s/recommendation", DEVICE_ID);
  snprintf(topicStatus, sizeof(topicStatus), "jitai/device/%s/status", DEVICE_ID);
}

void connectWifi() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("WiFi connecting");
  while (WiFi.status() != WL_CONNECTED) {
    delay(400);
    Serial.print(".");
  }
  Serial.println();
  Serial.print("WiFi OK ");
  Serial.println(WiFi.localIP());
}

void onMqttMessage(char* topic, byte* payload, unsigned int length) {
  Serial.print("MQTT reco [");
  Serial.print(topic);
  Serial.print("] ");
  for (unsigned int i = 0; i < length; i++) {
    Serial.print((char)payload[i]);
  }
  Serial.println();
}

bool connectMqtt() {
  mqtt.setServer(MQTT_HOST, MQTT_PORT);
  mqtt.setCallback(onMqttMessage);
  String clientId = String("jitai-") + DEVICE_ID + "-" + String((uint32_t)ESP.getEfuseMac(), HEX);
  bool ok;
#if defined(MQTT_USER) && defined(MQTT_PASSWORD)
  if (strlen(MQTT_USER) > 0) {
    ok = mqtt.connect(clientId.c_str(), MQTT_USER, MQTT_PASSWORD);
  } else {
    ok = mqtt.connect(clientId.c_str());
  }
#else
  ok = mqtt.connect(clientId.c_str());
#endif
  if (!ok) {
    Serial.print("MQTT failed rc=");
    Serial.println(mqtt.state());
    return false;
  }
  mqtt.subscribe(topicReco);
  mqtt.publish(topicStatus, "{\"online\":true}", true);
  Serial.println("MQTT OK");
  return true;
}

void publishTelemetry() {
  // Simple proxies: ramp steps, mild HR jitter. Swap for real sensors later.
  fakeSteps += 1.0f + (esp_random() % 30) / 10.0f;
  fakeHr = 70.0f + (esp_random() % 200) / 20.0f;

  StaticJsonDocument<384> doc;
  doc["device_id"] = DEVICE_ID;
  doc["seq"] = seq++;
  doc["t_ms"] = millis() - bootMs;
  doc["wifi_rssi"] = WiFi.RSSI();
  doc["steps"] = fakeSteps;
  doc["hr"] = fakeHr;
  doc["battery"] = 100;  // placeholder
  doc["schema"] = "jitai.telemetry.v1";

  char buf[384];
  size_t n = serializeJson(doc, buf, sizeof(buf));
  mqtt.publish(topicTelemetry, buf, n);
  Serial.println(buf);
}

void setup() {
  Serial.begin(115200);
  delay(200);
  bootMs = millis();
  setTopics();
  connectWifi();
  while (!connectMqtt()) {
    delay(1500);
  }
}

void loop() {
  if (WiFi.status() != WL_CONNECTED) {
    connectWifi();
  }
  if (!mqtt.connected()) {
    connectMqtt();
  }
  mqtt.loop();

  unsigned long now = millis();
  if (now - lastPublishMs >= (1000UL / TELEMETRY_HZ)) {
    lastPublishMs = now;
    publishTelemetry();
  }
}
