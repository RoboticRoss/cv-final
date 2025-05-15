#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SH110X.h>

Adafruit_SH1106G display = Adafruit_SH1106G(128, 64, &Wire);

int joyX1, joyY1;
int posX1 = 0, posY1 = 0;
int joyX2, joyY2;
int posX2 = 0, posY2 = 0;
int threshold = 400;

void setup() {
  display.begin(0x3C, true);
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SH110X_WHITE);
  posX1 = 0;
  posY1 = 0;
  posX2 = 0;
  posY2 = 0;
}

void loop() {
  joyX1 = analogRead(A0);
  joyY1 = analogRead(A1);

  joyX2 = analogRead(A2);
  joyY2 = analogRead(A3);

  // Basic deadzone
  if (joyX1 < threshold) posX1--;
  if (joyX1 > 1023 - threshold) posX1++;
  if (joyY1 < threshold) posY1++;
  if (joyY1 > 1023 - threshold) posY1--;

  if (joyX2 < threshold) posX2--;
  if (joyX2 > 1023 - threshold) posX2++;
  if (joyY2 < threshold) posY2++;
  if (joyY2 > 1023 - threshold) posY2--;

  posX1 = constrain(posX1, 0, 20); 
  posY1 = constrain(posY1, 0, 7); 
  
  posX2 = constrain(posX2, 0, 20); 
  posY2 = constrain(posY2, 0, 7); 
  display.clearDisplay();

  if (posX1 == posX2 && posY1 == posY2){
    display.setCursor(posX1 * 6, posY1 * 8);
    display.print("<3");
  }
  else {
    display.setCursor(posX1 * 6, posY1 * 8);
    display.print("R");
    display.setCursor(posX2 * 6, posY2 * 8);
    display.print("K");
  }

  display.display();

  delay(10); // Slow down the loop
}
