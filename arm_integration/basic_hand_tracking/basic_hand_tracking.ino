#include <Adafruit_GFX.h>
#include <Adafruit_SH110X.h>
#include <Wire.h>

#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET -1

Adafruit_SH1106G display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

int x = 0, y = 0, z = 0;
int debugX = 0, debugY = 0, debugZ = 0;

float smoothedX = 0;
float smoothedY = 0;
float smoothedZ = 0;
const float alpha = 0.92;

int mapRange(int val, int inMin, int inMax, int outMin, int outMax) {
  val = constrain(val, inMin, inMax);
  return (val - inMin) * (outMax - outMin) / (inMax - inMin) + outMin;
}

bool readSerial3D(int &xOut, int &yOut, int &zOut, int &xOut2, int &yOut2, int &zOut2) {
  if (!Serial.available()) return false;

  String input = Serial.readStringUntil('\n');

  int firstComma = input.indexOf(',');
  int secondComma = input.indexOf(',', firstComma + 1);

  if (firstComma > 0 && secondComma > firstComma) {
    String xStr = input.substring(0, firstComma);
    String yStr = input.substring(firstComma + 1, secondComma);
    String zStr = input.substring(secondComma + 1);

    int rawX = xStr.toInt();
    int rawY = yStr.toInt();
    int rawZ = zStr.toInt();
    xOut2=rawX;
    yOut2=rawY;
    zOut2=rawZ;


    // Ignore malformed input
    if (rawX < 0 || rawX > 255 ||
        rawY < 0 || rawY > 255 ||
        rawZ < 0 || rawZ > 254) {
      return false;  // reject noisy input
    }
    

    // Scale to display range
    smoothedX = alpha * smoothedX + (1.0 - alpha) * rawX;
    smoothedY = alpha * smoothedY + (1.0 - alpha) * rawY;
    smoothedZ = alpha * smoothedZ + (1.0 - alpha) * rawZ;

    // Use smoothed values for display
    int x = int(smoothedX);
    int y = int(smoothedY);
    int z = int(smoothedZ);

    xOut = mapRange(x, 0, 255, 0, 127);
    yOut = mapRange(y, 0, 255, 0, 57);
    zOut = z;


    return true;
  }

  return false;
}

void setup() {
  Serial.begin(9600);
  display.begin(0x3C, true);
  display.clearDisplay();
  display.setTextColor(SH110X_WHITE);
  display.setTextSize(1);
}

void draw3DPoint(int x, int y, int z) {
  x = constrain(x, 0, 255);
  y = constrain(y, 0, 255);
  z = constrain(z, 1, 255); 

  float scale = 64.0 / z; // lower z = bigger
  int centerX = SCREEN_WIDTH / 2;
  int centerY = SCREEN_HEIGHT / 2;

  int screenX = 127-x;
  int screenY = y;

  int radius = max(2, int(9 - z / 30));  // closer = bigger dot

  display.fillCircle(screenX, screenY, radius, SH110X_WHITE);
}

void loop() {
  if (readSerial3D(x, y, z, debugX, debugY, debugZ)) {

    display.clearDisplay();
    draw3DPoint(x, y, z);

    display.setCursor(0, SCREEN_HEIGHT - 12);
    display.setTextSize(1);
    display.print("X: "); display.print(x);
    display.print(" Y: "); display.print(y);
    display.print(" Z: "); display.print(z);
    // display.print("\nRAWX: "); display.print(debugX);
    // display.print("\nRAWY: "); display.print(debugY);
    // display.print("\nRAWZ: "); display.print(debugZ);

    display.display();
  }

  delay(10); // smooth display
}
