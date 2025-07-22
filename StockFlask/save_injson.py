#save injson
import json

access_token = "4IhaK00ucUT3X7jin8Rrdsbr7vNGDRaO"

# Save to JSON
with open("access_token.json", "w") as f:
    json.dump({"access_token": access_token}, f, indent=2)
    print("💾 Access token saved to access_token.json")
