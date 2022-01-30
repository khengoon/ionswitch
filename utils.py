import requests

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_ion = load_lottieurl('https://assets3.lottiefiles.com/packages/lf20_wpGbRj.json')