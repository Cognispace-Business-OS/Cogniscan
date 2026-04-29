import json

# Load the Cookie-Editor export (list of cookie objects)
def cookie_converter():
    with open("cookies.json", "r") as f:
        raw = json.load(f)

    # Convert to simple {name: value} dict that twikit expects
    converted = {cookie["name"]: cookie["value"] for cookie in raw}

    with open("cookies.json", "w") as f:
        json.dump(converted, f, indent=2)

    print("✅ cookies.json converted successfully")
    print("Keys found:", list(converted.keys()))

if __name__ == "__main__":
    cookie_converter()