# Android Friend Labeler

This project includes a native Android APK for friend labeling. It packages the current image manifest into the app, stores labels locally, and exports a JSON label bundle through Android's share/mail sheet.

## Build the APK

```powershell
powershell -ExecutionPolicy Bypass -File D:\CloudAI\cloud-aesthetics\scripts\build_friend_labeler_apk.ps1 `
  -OutputApk D:\CloudAI\cloud-aesthetics\data\artifacts\cloud_friend_labeler.apk `
  -PackageName cloud_mobile_labeler `
  -RecipientEmail "your-mail@example.com"
```

If `-RecipientEmail` is omitted, friends can still enter the target email in the app before exporting.

The default build optimizes images for phones (`MaxImageEdge=1600`, `JpegQuality=88`). The labels still use the original manifest `image_id` values.

The native app keeps the UI in `MainActivity.java` and the reusable sky/cloud crop heuristic in `CloudCropper.java`, so the crop logic can be improved without touching the phone workflow.

## Friend Flow

1. Send `data/artifacts/cloud_friend_labeler.apk`.
2. Friend installs it on Android and opens `Cloud Labeler`.
3. They enter a rater name, label images, optionally use Duel mode, and can import their own images.
4. They tap `Mail` or `Export`, then send the generated JSON attachment.

Android does not allow silent background email sending without account credentials. The app creates the email/share intent with the JSON attachment; the friend confirms the send in their mail app.

## Import Returned Labels

```powershell
D:\CloudAI\.venv\Scripts\python.exe -m cloud_aesthetics.cli import-friend-labels --bundle D:\Path\To\cloud_mobile_labeler_friend_labels.json
D:\CloudAI\.venv\Scripts\python.exe -m cloud_aesthetics.cli ingest-images
D:\CloudAI\.venv\Scripts\python.exe -m cloud_aesthetics.cli aggregate-labels
```

Imported friend images are restored under `data/raw/images/friend_imports`. The importer remaps embedded image IDs to the SHA-based IDs that the local manifest will calculate.
