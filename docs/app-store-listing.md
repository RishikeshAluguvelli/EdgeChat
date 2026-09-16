# App Store submission (copy-paste)

Everything you need to ship the iOS app through App Store Connect. The Android equivalent is in `play-listing.md`.

## Prerequisites (one-time)

1. **Apple Developer Program** membership (US$99/year) at https://developer.apple.com/programs/enroll/. A free Apple ID can build to your own phone but cannot use TestFlight or App Store Connect. Enrolment takes from a few hours to ~2 days.
2. Xcode → Settings → Accounts: sign in with that Apple ID once enrolment is approved. Team `4J6HRDQT39` is already set in `App/project.yml`, so automatic signing picks up the new App Store distribution certificate and profile by itself.
3. App Store Connect → Users and Access → Agreements: accept the **Paid Apps** agreement only if you ever add purchases; the free-app agreement is accepted during enrolment.

## Register the app

1. https://appstoreconnect.apple.com → My Apps → **+** → New App.
   Platform iOS, name `EdgeChat`, primary language English (U.S.), bundle ID `com.rishikesh.edgechat` (create it under Certificates, Identifiers & Profiles if it is not offered), SKU `edgechat-ios`, full access.
2. If `EdgeChat` is taken as a name, use `EdgeChat – Offline AI Chat` (the subtitle field then stays free).

## Upload a build

From the repo root:

```bash
cd App
xcodegen generate                     # only if project.yml changed
open EdgeChat.xcodeproj
```

In Xcode: select **Any iOS Device (arm64)** as the run destination → Product → **Archive** → Organizer opens → **Distribute App** → **App Store Connect** → Upload (keep "Upload your app's symbols" and "Manage version and build number" on). Command-line equivalent:

```bash
xcodebuild -project EdgeChat.xcodeproj -scheme EdgeChat -configuration Release \
  -destination 'generic/platform=iOS' -archivePath build/EdgeChat.xcarchive \
  -allowProvisioningUpdates archive
xcodebuild -exportArchive -archivePath build/EdgeChat.xcarchive \
  -exportOptionsPlist ExportOptions.plist -exportPath build/export -allowProvisioningUpdates
# ExportOptions.plist: {method: app-store-connect, teamID: 4J6HRDQT39, uploadSymbols: true}
```

Bump `MARKETING_VERSION` (user-visible, e.g. 1.0.0) and `CURRENT_PROJECT_VERSION` (build number, must increase for every upload) in `App/project.yml`, then `xcodegen generate`. The first App Store release should be 1.0.0 / build 1.

Processing takes 5–30 minutes; you get an email when the build is usable. Export compliance is pre-answered by `ITSAppUsesNonExemptEncryption = false` in Info.plist (the app only uses HTTPS for model downloads), so no encryption questionnaire appears.

## TestFlight (optional but recommended first)

TestFlight → the processed build → add yourself to an internal group (no review needed, installs immediately) or create an external group (a short beta review, then a public link for up to 10,000 testers). This is the easiest way to give the README a "Try it on iPhone" link before the store review is done.

## App Store listing

**Name** (30): `EdgeChat`
**Subtitle** (30): `Private AI chat, fully offline`
**Category**: Productivity (secondary: Utilities)
**Promotional text** (170, editable without a new build):
`Runs entirely on your iPhone. Download a model once, then chat, ask about photos and read PDFs with no internet and no account.`

**Description** (4000):

```
EdgeChat is a ChatGPT-style assistant that runs completely on your iPhone. Download a model once, then chat, ask questions about photos and read PDFs with no internet connection and no account.

WHY EDGECHAT
• Fully offline: replies are generated on your device with llama.cpp and Metal. Airplane mode works.
• Private by design: no server, no analytics, no telemetry. Your conversations never leave the phone.
• Photos and documents: attach a photo and ask about it (vision models), or attach a PDF, text or code file and ask questions about its contents. Scanned pages are read with on-device OCR.
• Long conversations that stay coherent: EdgeChat keeps a running memory of older messages, recalls relevant earlier passages, and continues long replies instead of cutting them off.
• Choose your model: Qwen3-VL 4B and 2B (with vision), Qwen3 4B and 1.7B, Gemma 3 4B, Llama 3.2 3B, and any GGUF file you import through Files.
• Honest stats: every reply shows tokens per second, context usage and why it stopped.

WHAT YOU NEED
iPhone 12 or newer with iOS 17. 4B models are best on iPhone 15 Pro and later (8 GB RAM); on 6 GB phones use the 2B / 1.7B tier. Model files are 0.4–3 GB and are downloaded once over Wi-Fi.

OPEN SOURCE
EdgeChat is MIT-licensed: github.com/RishikeshAluguvelli/EdgeChat
```

**Keywords** (100): `offline ai,local llm,private chat,llama,qwen,gemma,on-device,chatbot,gguf,pdf chat`
**Support URL**: `https://github.com/RishikeshAluguvelli/EdgeChat/issues`
**Marketing URL**: `https://github.com/RishikeshAluguvelli/EdgeChat`
**Privacy Policy URL**: `https://github.com/RishikeshAluguvelli/EdgeChat/blob/main/docs/privacy-policy.md`
**Copyright**: `2026 Rishikesh Reddy Aluguvelli`

### Screenshots

Required sizes: **6.9" iPhone** (1320×2868, from an iPhone 16 Pro Max simulator) and, because the target supports iPad, **13" iPad** (2064×2752, iPad Pro 13" simulator). The README screenshots in `docs/screenshots/` are 553×1200 and will be rejected; regenerate them at full resolution:

```bash
xcrun simctl boot "iPhone 16 Pro Max"
xcrun simctl launch booted com.rishikesh.edgechat
xcrun simctl io booted screenshot chat.png      # repeat for chats, models, settings
```

Use 3–6 shots: a chat with a long answer, the photo question, the Models screen, Settings, the context ring. No device frames are required.

### App privacy (nutrition label)

- Data collection: **No, we do not collect data from this app.**
  (Everything is on-device; the only network request is the model download from huggingface.co, which contains no user data. This matches `App/EdgeChat/PrivacyInfo.xcprivacy`, which declares no tracking and no collected data.)

### Age rating

Answer the questionnaire honestly (no violence, no gambling, no unrestricted web). Then, under the generative-AI question, state that the app generates text and image descriptions from a user-chosen open model **without server-side filtering**. Apple usually requires **17+ / 18+** for unfiltered generative chat; set it up front rather than after a rejection.

### App Review information

- Sign-in required: **No**.
- Notes for the reviewer:

```
EdgeChat runs language models locally; there is no server. To test quickly: tap Models, download "Qwen3 0.6B" (about 400 MB, 1 minute on Wi-Fi), tap Load, then send a message. Vision: download "Qwen3-VL 2B", attach a photo with the paperclip. Files: attach any PDF. Everything works in Airplane Mode after the download. The increased-memory-limit entitlement is used because 4B models need ~3 GB of RAM.
```

- Contact: your phone number and email (required).

## Submit

Version page → select the build → Save → **Add for Review** → **Submit to App Review**. First reviews take 24–48 hours. Typical follow-ups for this kind of app and how to answer them:

| Reviewer asks | Answer |
| --- | --- |
| Where is the content generated? | On-device (llama.cpp with Metal); attach a screenshot of the reply stats showing tokens/s. |
| Age rating for AI content | Accept 17+/18+. |
| Guideline 2.1 crash on iPad | Test on an iPad simulator once; the app supports both device families. If you would rather not maintain iPad, set `TARGETED_DEVICE_FAMILY: "1"` in `project.yml` before uploading. |
| Guideline 5.1.1 photo permission | The purpose strings in Info.plist already say nothing leaves the phone. |

Once approved, the store link is `https://apps.apple.com/app/id<APPLE_ID>` (Apple ID number shown in App Information); add it to the README "Try it" section next to the Play link.

## Updates

Every update: bump version + build in `project.yml`, `xcodegen generate`, archive, upload, add "What's New", submit. Phased release (7 days) is on by default for updates.
