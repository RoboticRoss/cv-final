void setup() {
  Serial.begin(9600);    
  pinMode(13, OUTPUT);
}

void loop() {
  if (Serial.available() > 0) {
    char cmd = Serial.read();
    if (cmd == '1') {
      digitalWrite(13, HIGH); //LED on
    } else if (cmd == '0') {
      digitalWrite(13, LOW); //LED off
    }
  }
}