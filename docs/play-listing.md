# Google Play listing (copy-paste)

Everything below is ready to paste into Play Console → Grow → Store presence → Main store listing.

**App name** (30): `EdgeChat – Offline AI Chat`

**Short description** (80):
`Private AI chat that runs entirely on your phone. No account, no server, works offline.`

**Full description** (4000):

```
EdgeChat is a ChatGPT-style assistant that runs completely on your phone. Download a model once, then chat, ask questions about photos and read PDFs with no internet connection and no account.

WHY EDGECHAT
• Fully offline: replies are generated on your device with llama.cpp. Airplane mode works.
• Private by design: no server, no analytics, no telemetry. Your conversations never leave the phone.
• Photos and documents: attach a photo and ask about it (vision models), or attach a PDF, text or code file and ask questions about its contents. Scanned pages are read with on-device OCR.
• Long conversations that stay coherent: EdgeChat keeps a running memory of older messages, recalls relevant earlier passages, and continues long replies instead of cutting them off.
• Choose your model: Qwen3-VL 4B and 2B (with vision), Qwen3 4B and 1.7B, Gemma 3 4B, Llama 3.2 3B, and any GGUF file you import.
• Honest stats: every reply shows tokens per second, context usage and why it stopped.

WHAT YOU NEED
A phone with at least 6 GB of RAM. 4B models are best on 8 GB+ devices; 6 GB phones should use the 2B / 1.7B tier. Model files are 0.4–3 GB and are downloaded once over Wi-Fi.

OPEN SOURCE
EdgeChat is MIT-licensed: github.com/RishikeshAluguvelli/EdgeChat
```

**App category**: Productivity. **Tags**: AI assistant, chatbot, offline.

**Contact email**: your address. **Privacy policy URL**: `https://github.com/RishikeshAluguvelli/EdgeChat/blob/main/docs/privacy-policy.md`
(or publish `docs/privacy-policy.md` on your GitHub Pages site and use that URL).

## Data safety form

- Does your app collect or share any of the required user data types? **No.**
- Is all of the user data collected by your app encrypted in transit? N/A (no collection).
- Do you provide a way for users to request that their data is deleted? N/A.

The only network traffic is the model download from huggingface.co, which carries no user data.

## Content rating questionnaire

Category: Utility / Productivity. No user-generated content is shared with other users; no violence, sexual content, gambling, or purchases. Result should be "Everyone".

## App access

All features are available without login. Note for reviewers: "Tap Models, download 'Qwen3 0.6B' (400 MB) to try the app quickly; larger models give better answers."

## Release checklist

1. Play Console → Create app (default language en-US, app, free).
2. Set up → App content: privacy policy URL, ads (No), app access (all functionality available), content rating, target audience (18+ or 13+), data safety (no data collected).
3. Release → Testing → Internal testing → Create release → upload `android/app/build/outputs/bundle/release/app-release.aab`. Play App Signing: accept (the `upload-keystore.jks` in `android/` is the **upload** key; back it up, it is not in git).
4. Add testers by email, roll out. The internal-test link installs immediately.
5. Production requires (for personal accounts created after Nov 2023) a closed test with at least 12 testers opted in for 14 days before you can apply for production access.
6. Store listing: paste the texts above, add the icon (`android/app/src/main/res/mipmap-xxxhdpi/ic_launcher_foreground.png` at 512×512) and at least two phone screenshots (`docs/screenshots/android-*.png`) and a 1024×500 feature graphic (`docs/screenshots/hero.png` cropped).
7. Bump `versionCode` in `android/app/build.gradle.kts` for every upload.
