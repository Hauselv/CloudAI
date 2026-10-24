package com.cloudaesthetics.friendlabeler;

import android.app.Activity;
import android.content.ClipData;
import android.content.Intent;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.graphics.drawable.GradientDrawable;
import android.net.Uri;
import android.os.Bundle;
import android.provider.OpenableColumns;
import android.text.InputType;
import android.util.Base64;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.EditText;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.ScrollView;
import android.widget.SeekBar;
import android.widget.TextView;
import android.widget.Toast;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStreamWriter;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashSet;
import java.util.Locale;
import java.util.Random;
import java.util.Set;

public class MainActivity extends Activity {
    private static final int BG = 0xFF101418;
    private static final int PANEL = 0xFF171D24;
    private static final int PANEL_2 = 0xFF202833;
    private static final int FG = 0xFFF5F7FA;
    private static final int MUTED = 0xFF9AA4B2;
    private static final int ACCENT = 0xFF4EA3FF;
    private static final int GOOD = 0xFF33D17A;
    private static final int WARNING = 0xFFFFC857;
    private static final int REQUEST_IMPORT_IMAGES = 4021;

    private final ArrayList<ImageItem> packagedImages = new ArrayList<>();
    private final ArrayList<ImageItem> importedImages = new ArrayList<>();
    private final Random random = new Random();

    private JSONArray ratings = new JSONArray();
    private JSONArray pairwise = new JSONArray();
    private LinearLayout root;
    private FrameLayout contentFrame;
    private TextView statText;
    private TextView progressText;
    private ProgressBar progressBar;
    private EditText raterEdit;
    private String packageName = "cloud_mobile_labeler";
    private String recipientEmail = "";
    private String mode = "rating";
    private int currentIndex = 0;
    private int streak = 0;
    private int lastLevel = 1;
    private int[] currentPair = null;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        try {
            loadManifest();
            loadState();
            buildShell();
            render();
        } catch (Exception error) {
            TextView failed = new TextView(this);
            failed.setTextColor(FG);
            failed.setTextSize(16);
            failed.setPadding(dp(18), dp(18), dp(18), dp(18));
            failed.setText("App konnte nicht gestartet werden:\n" + error.getMessage());
            setContentView(failed);
        }
    }

    private void loadManifest() throws IOException, JSONException {
        JSONObject manifest = new JSONObject(readText(getAssets().open("manifest.json")));
        packageName = manifest.optString("package_name", packageName);
        recipientEmail = manifest.optString("recipient_email", "");
        JSONArray images = manifest.getJSONArray("images");
        for (int i = 0; i < images.length(); i++) {
            JSONObject row = images.getJSONObject(i);
            ImageItem item = new ImageItem();
            item.imageId = row.getString("image_id");
            item.assetPath = row.getString("path");
            item.originalPath = row.optString("original_relative_path", item.assetPath);
            item.width = row.optInt("width", 0);
            item.height = row.optInt("height", 0);
            item.imported = false;
            packagedImages.add(item);
        }
    }

    private void loadState() throws JSONException {
        android.content.SharedPreferences prefs = getSharedPreferences("cloud_labeler", MODE_PRIVATE);
        ratings = new JSONArray(prefs.getString("ratings", "[]"));
        pairwise = new JSONArray(prefs.getString("pairwise", "[]"));
        currentIndex = prefs.getInt("current_index", 0);
        streak = prefs.getInt("streak", 0);
        lastLevel = prefs.getInt("last_level", 1);
        recipientEmail = prefs.getString("recipient_email", recipientEmail);
        JSONArray imported = new JSONArray(prefs.getString("imported_images", "[]"));
        for (int i = 0; i < imported.length(); i++) {
            JSONObject row = imported.getJSONObject(i);
            File file = new File(row.getString("file_path"));
            if (!file.exists()) {
                continue;
            }
            ImageItem item = new ImageItem();
            item.imageId = row.getString("image_id");
            item.filePath = file.getAbsolutePath();
            item.originalPath = row.optString("original_relative_path", file.getName());
            item.width = row.optInt("width", 0);
            item.height = row.optInt("height", 0);
            item.imported = true;
            importedImages.add(item);
        }
    }

    private void saveState() {
        getSharedPreferences("cloud_labeler", MODE_PRIVATE)
            .edit()
            .putString("ratings", ratings.toString())
            .putString("pairwise", pairwise.toString())
            .putString("imported_images", importedMetadata().toString())
            .putInt("current_index", currentIndex)
            .putInt("streak", streak)
            .putInt("last_level", lastLevel)
            .putString("recipient_email", recipientEmail)
            .apply();
    }

    private JSONArray importedMetadata() {
        JSONArray rows = new JSONArray();
        for (ImageItem item : importedImages) {
            JSONObject row = new JSONObject();
            try {
                row.put("image_id", item.imageId);
                row.put("file_path", item.filePath);
                row.put("original_relative_path", item.originalPath);
                row.put("width", item.width);
                row.put("height", item.height);
                rows.put(row);
            } catch (JSONException ignored) {
            }
        }
        return rows;
    }

    private void buildShell() {
        root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setBackgroundColor(BG);
        root.setPadding(dp(12), dp(10) + systemBarHeight("status_bar_height"), dp(12), dp(10) + systemBarHeight("navigation_bar_height"));

        LinearLayout header = new LinearLayout(this);
        header.setGravity(Gravity.CENTER_VERTICAL);
        header.setOrientation(LinearLayout.HORIZONTAL);
        root.addView(header, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(58)));

        LinearLayout titleBox = new LinearLayout(this);
        titleBox.setOrientation(LinearLayout.VERTICAL);
        TextView title = new TextView(this);
        title.setText("Cloud Labeler");
        title.setTextColor(FG);
        title.setTextSize(22);
        title.setTypeface(android.graphics.Typeface.DEFAULT_BOLD);
        statText = new TextView(this);
        statText.setTextColor(MUTED);
        statText.setTextSize(12);
        titleBox.addView(title);
        titleBox.addView(statText);
        header.addView(titleBox, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1));

        raterEdit = new EditText(this);
        raterEdit.setSingleLine(true);
        raterEdit.setText(getSharedPreferences("cloud_labeler", MODE_PRIVATE).getString("rater_id", "friend"));
        raterEdit.setTextColor(FG);
        raterEdit.setHintTextColor(MUTED);
        raterEdit.setHint("Rater");
        raterEdit.setTextSize(13);
        raterEdit.setSelectAllOnFocus(true);
        raterEdit.setBackground(panelDrawable(PANEL_2, dp(8), 0));
        raterEdit.setPadding(dp(10), 0, dp(10), 0);
        header.addView(raterEdit, new LinearLayout.LayoutParams(dp(96), dp(42)));

        Button export = button("Mail", true);
        export.setOnClickListener(v -> exportLabelsByMail());
        LinearLayout.LayoutParams exportParams = new LinearLayout.LayoutParams(dp(76), dp(42));
        exportParams.leftMargin = dp(8);
        header.addView(export, exportParams);

        LinearLayout modes = new LinearLayout(this);
        modes.setOrientation(LinearLayout.HORIZONTAL);
        modes.setGravity(Gravity.CENTER);
        root.addView(modes, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(46)));
        addModeButton(modes, "Rating", "rating");
        addModeButton(modes, "Duel", "duel");
        addModeButton(modes, "Import", "import");
        addModeButton(modes, "Export", "export");

        progressText = new TextView(this);
        progressText.setTextColor(MUTED);
        progressText.setTextSize(13);
        root.addView(progressText, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(24)));

        progressBar = new ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal);
        progressBar.setMax(1000);
        root.addView(progressBar, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(8)));

        contentFrame = new FrameLayout(this);
        LinearLayout.LayoutParams frameParams = new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1);
        frameParams.topMargin = dp(10);
        root.addView(contentFrame, frameParams);
        setContentView(root);
    }

    private void addModeButton(LinearLayout modes, String label, String targetMode) {
        Button item = button(label, false);
        item.setOnClickListener(v -> {
            mode = targetMode;
            render();
        });
        LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(0, dp(38), 1);
        params.leftMargin = dp(3);
        params.rightMargin = dp(3);
        modes.addView(item, params);
    }

    private void render() {
        updateStats();
        contentFrame.removeAllViews();
        if ("duel".equals(mode)) {
            renderDuel();
        } else if ("import".equals(mode)) {
            renderImport();
        } else if ("export".equals(mode)) {
            renderExport();
        } else {
            renderRating();
        }
    }

    private void updateStats() {
        int total = allImages().size();
        int rated = ratedIds().size();
        int xp = ratings.length() * 10 + pairwise.length() * 4;
        int level = xp / 100 + 1;
        statText.setText("Level " + level + " | XP " + xp + " | Streak " + streak);
        progressText.setText("Fortschritt: " + rated + " / " + total + " Bilder");
        progressBar.setProgress(total == 0 ? 0 : Math.round((rated * 1000f) / total));
        if (level > lastLevel) {
            lastLevel = level;
            toast("Level up: " + level);
        }
    }

    private void renderRating() {
        ArrayList<ImageItem> images = allImages();
        if (images.isEmpty()) {
            contentFrame.addView(emptyText("Keine Bilder im Paket."));
            return;
        }
        currentIndex = Math.max(0, Math.min(currentIndex, images.size() - 1));
        ImageItem item = images.get(currentIndex);

        LinearLayout panel = new LinearLayout(this);
        panel.setOrientation(LinearLayout.VERTICAL);
        panel.setGravity(Gravity.CENTER_HORIZONTAL);
        contentFrame.addView(panel, new FrameLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT));

        ImageView image = new ImageView(this);
        image.setBackgroundColor(0xFF090B0E);
        image.setScaleType(ImageView.ScaleType.FIT_CENTER);
        image.setImageBitmap(loadBitmap(item, 1800));
        panel.addView(image, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1));

        TextView caption = new TextView(this);
        caption.setTextColor(MUTED);
        caption.setTextSize(12);
        caption.setSingleLine(true);
        caption.setText((currentIndex + 1) + " / " + images.size() + "  " + item.imageId + "  " + item.originalPath);
        panel.addView(caption, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(28)));

        TextView scoreLabel = new TextView(this);
        scoreLabel.setTextColor(FG);
        scoreLabel.setTextSize(17);
        scoreLabel.setTypeface(android.graphics.Typeface.DEFAULT_BOLD);
        panel.addView(scoreLabel, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(30)));

        SeekBar score = new SeekBar(this);
        score.setMax(20);
        double existingScore = existingRatingScore(item.imageId, 8.0);
        score.setProgress((int) Math.round(existingScore * 2.0));
        scoreLabel.setText("Score: " + formatScore(score.getProgress() / 2.0));
        score.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                scoreLabel.setText("Score: " + formatScore(progress / 2.0));
            }
            @Override public void onStartTrackingTouch(SeekBar seekBar) {}
            @Override public void onStopTrackingTouch(SeekBar seekBar) {}
        });
        panel.addView(score, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(44)));

        LinearLayout quick = new LinearLayout(this);
        quick.setOrientation(LinearLayout.HORIZONTAL);
        panel.addView(quick, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(40)));
        addScoreButton(quick, "3", score, 6);
        addScoreButton(quick, "5", score, 10);
        addScoreButton(quick, "7", score, 14);
        addScoreButton(quick, "9", score, 18);
        addScoreButton(quick, "10", score, 20);

        EditText note = new EditText(this);
        note.setSingleLine(true);
        note.setText(existingRatingNote(item.imageId));
        note.setTextColor(FG);
        note.setHintTextColor(MUTED);
        note.setHint("Notiz optional");
        note.setBackground(panelDrawable(PANEL, dp(8), 0));
        note.setPadding(dp(10), 0, dp(10), 0);
        panel.addView(note, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(44)));

        LinearLayout actions = new LinearLayout(this);
        actions.setOrientation(LinearLayout.HORIZONTAL);
        LinearLayout.LayoutParams actionParams = new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(52));
        actionParams.topMargin = dp(8);
        panel.addView(actions, actionParams);

        Button previous = button("Zurueck", false);
        previous.setOnClickListener(v -> {
            currentIndex = Math.max(0, currentIndex - 1);
            saveState();
            render();
        });
        actions.addView(previous, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.MATCH_PARENT, 1));

        Button next = button("Weiter", false);
        next.setOnClickListener(v -> {
            currentIndex = Math.min(allImages().size() - 1, currentIndex + 1);
            saveState();
            render();
        });
        LinearLayout.LayoutParams nextParams = new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.MATCH_PARENT, 1);
        nextParams.leftMargin = dp(8);
        actions.addView(next, nextParams);

        Button save = button("Speichern + next", true);
        save.setOnClickListener(v -> saveRating(item, score.getProgress() / 2.0, note.getText().toString()));
        LinearLayout.LayoutParams saveParams = new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.MATCH_PARENT, 1.6f);
        saveParams.leftMargin = dp(8);
        actions.addView(save, saveParams);
    }

    private void addScoreButton(LinearLayout parent, String label, SeekBar seekBar, int progress) {
        Button item = button(label, false);
        item.setOnClickListener(v -> seekBar.setProgress(progress));
        LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.MATCH_PARENT, 1);
        params.leftMargin = dp(3);
        params.rightMargin = dp(3);
        parent.addView(item, params);
    }

    private void saveRating(ImageItem item, double score, String note) {
        persistRater();
        JSONArray next = new JSONArray();
        for (int i = 0; i < ratings.length(); i++) {
            JSONObject row = ratings.optJSONObject(i);
            if (row == null || !item.imageId.equals(row.optString("image_id"))) {
                next.put(row);
            }
        }
        JSONObject row = new JSONObject();
        try {
            row.put("image_id", item.imageId);
            row.put("rater_id", raterEdit.getText().toString().trim());
            row.put("rating_session_id", sessionId());
            row.put("score", score);
            row.put("note", note);
            row.put("timestamp", isoNow());
            next.put(row);
        } catch (JSONException ignored) {
        }
        ratings = next;
        streak += 1;
        if (streak > 0 && streak % 10 == 0) {
            toast(streak + "er Streak");
        }
        nextUnrated();
        saveState();
        render();
    }

    private void nextUnrated() {
        ArrayList<ImageItem> images = allImages();
        Set<String> rated = ratedIds();
        for (int offset = 1; offset <= images.size(); offset++) {
            int candidate = (currentIndex + offset) % images.size();
            if (!rated.contains(images.get(candidate).imageId)) {
                currentIndex = candidate;
                return;
            }
        }
        if (!images.isEmpty()) {
            currentIndex = (currentIndex + 1) % images.size();
        }
    }

    private void renderDuel() {
        ArrayList<ImageItem> images = allImages();
        if (images.size() < 2) {
            contentFrame.addView(emptyText("Mindestens zwei Bilder noetig."));
            return;
        }
        if (currentPair == null || currentPair[0] >= images.size() || currentPair[1] >= images.size()) {
            pickPair(images.size());
        }
        ImageItem left = images.get(currentPair[0]);
        ImageItem right = images.get(currentPair[1]);

        LinearLayout panel = new LinearLayout(this);
        panel.setOrientation(LinearLayout.VERTICAL);
        contentFrame.addView(panel, new FrameLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT));

        LinearLayout duel = new LinearLayout(this);
        duel.setOrientation(LinearLayout.HORIZONTAL);
        panel.addView(duel, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1));
        duel.addView(duelImage(left), new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.MATCH_PARENT, 1));
        LinearLayout.LayoutParams rightParams = new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.MATCH_PARENT, 1);
        rightParams.leftMargin = dp(8);
        duel.addView(duelImage(right), rightParams);

        TextView hint = new TextView(this);
        hint.setTextColor(MUTED);
        hint.setTextSize(13);
        hint.setText("Welches Bild ist staerker?");
        hint.setGravity(Gravity.CENTER);
        panel.addView(hint, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(34)));

        LinearLayout actions = new LinearLayout(this);
        actions.setOrientation(LinearLayout.HORIZONTAL);
        panel.addView(actions, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(56)));
        Button leftButton = button("Links", true);
        leftButton.setOnClickListener(v -> savePairwise(left, right, left.imageId, false));
        actions.addView(leftButton, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.MATCH_PARENT, 1));
        Button tieButton = button("Tie", false);
        tieButton.setOnClickListener(v -> savePairwise(left, right, null, true));
        LinearLayout.LayoutParams tieParams = new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.MATCH_PARENT, 1);
        tieParams.leftMargin = dp(8);
        actions.addView(tieButton, tieParams);
        Button rightButton = button("Rechts", true);
        rightButton.setOnClickListener(v -> savePairwise(left, right, right.imageId, false));
        LinearLayout.LayoutParams rightButtonParams = new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.MATCH_PARENT, 1);
        rightButtonParams.leftMargin = dp(8);
        actions.addView(rightButton, rightButtonParams);
    }

    private View duelImage(ImageItem item) {
        LinearLayout box = new LinearLayout(this);
        box.setOrientation(LinearLayout.VERTICAL);
        box.setBackground(panelDrawable(PANEL, dp(8), 0));
        box.setPadding(dp(6), dp(6), dp(6), dp(6));
        ImageView image = new ImageView(this);
        image.setScaleType(ImageView.ScaleType.FIT_CENTER);
        image.setBackgroundColor(0xFF090B0E);
        image.setImageBitmap(loadBitmap(item, 1200));
        box.addView(image, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1));
        TextView label = new TextView(this);
        label.setTextColor(MUTED);
        label.setTextSize(11);
        label.setSingleLine(true);
        label.setText(item.imageId);
        box.addView(label, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(24)));
        return box;
    }

    private void pickPair(int size) {
        int left = random.nextInt(size);
        int right = random.nextInt(size);
        while (right == left) {
            right = random.nextInt(size);
        }
        currentPair = new int[] {left, right};
    }

    private void savePairwise(ImageItem left, ImageItem right, String winner, boolean tie) {
        persistRater();
        JSONObject row = new JSONObject();
        try {
            row.put("left_image_id", left.imageId);
            row.put("right_image_id", right.imageId);
            row.put("rater_id", raterEdit.getText().toString().trim());
            row.put("winner", winner == null ? JSONObject.NULL : winner);
            row.put("tie_flag", tie);
            row.put("preference_strength", tie ? 0.5 : 1.0);
            row.put("timestamp", isoNow());
            pairwise.put(row);
        } catch (JSONException ignored) {
        }
        streak += 1;
        pickPair(allImages().size());
        saveState();
        render();
    }

    private void renderImport() {
        ScrollView scroll = new ScrollView(this);
        LinearLayout panel = new LinearLayout(this);
        panel.setOrientation(LinearLayout.VERTICAL);
        panel.setPadding(0, dp(8), 0, dp(8));
        scroll.addView(panel);
        contentFrame.addView(scroll);

        TextView title = heading("Eigene Bilder importieren");
        panel.addView(title);
        TextView copy = body("Die App erstellt lokal Originale und bis zu 8 Sky/Cloud-Crops pro Bild. Beim Mail-Export werden diese Bilder eingebettet.");
        panel.addView(copy);

        Button choose = button("Bilder auswaehlen", true);
        choose.setOnClickListener(v -> openImagePicker());
        LinearLayout.LayoutParams chooseParams = new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(54));
        chooseParams.topMargin = dp(14);
        panel.addView(choose, chooseParams);

        Button clear = button("Importe loeschen", false);
        clear.setOnClickListener(v -> {
            deleteImportedImages();
            saveState();
            render();
        });
        LinearLayout.LayoutParams clearParams = new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(48));
        clearParams.topMargin = dp(10);
        panel.addView(clear, clearParams);

        TextView count = body("Importierte Bilder/Crops: " + importedImages.size());
        LinearLayout.LayoutParams countParams = new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        countParams.topMargin = dp(18);
        panel.addView(count, countParams);
    }

    private void renderExport() {
        ScrollView scroll = new ScrollView(this);
        LinearLayout panel = new LinearLayout(this);
        panel.setOrientation(LinearLayout.VERTICAL);
        panel.setPadding(0, dp(8), 0, dp(8));
        scroll.addView(panel);
        contentFrame.addView(scroll);

        panel.addView(heading("Export per Mail"));
        panel.addView(body("Die App erzeugt eine JSON-Datei mit Ratings, Duellen und importierten Bildern. Android oeffnet danach die Mail-App mit Anhang."));

        EditText email = new EditText(this);
        email.setSingleLine(true);
        email.setInputType(InputType.TYPE_TEXT_VARIATION_EMAIL_ADDRESS);
        email.setText(recipientEmail);
        email.setHint("Empfaenger-Mail");
        email.setTextColor(FG);
        email.setHintTextColor(MUTED);
        email.setBackground(panelDrawable(PANEL, dp(8), 0));
        email.setPadding(dp(12), 0, dp(12), 0);
        LinearLayout.LayoutParams emailParams = new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(52));
        emailParams.topMargin = dp(14);
        panel.addView(email, emailParams);

        Button send = button("JSON exportieren und Mail oeffnen", true);
        send.setOnClickListener(v -> {
            recipientEmail = email.getText().toString().trim();
            saveState();
            exportLabelsByMail();
        });
        LinearLayout.LayoutParams sendParams = new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(56));
        sendParams.topMargin = dp(12);
        panel.addView(send, sendParams);

        panel.addView(body("Aktuell: " + ratings.length() + " Ratings, " + pairwise.length() + " Duelle, " + importedImages.size() + " importierte Bilder/Crops."));
    }

    private void openImagePicker() {
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("image/*");
        intent.putExtra(Intent.EXTRA_ALLOW_MULTIPLE, true);
        startActivityForResult(intent, REQUEST_IMPORT_IMAGES);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode != REQUEST_IMPORT_IMAGES || resultCode != RESULT_OK || data == null) {
            return;
        }
        ArrayList<Uri> uris = new ArrayList<>();
        ClipData clipData = data.getClipData();
        if (clipData != null) {
            for (int i = 0; i < clipData.getItemCount(); i++) {
                uris.add(clipData.getItemAt(i).getUri());
            }
        } else if (data.getData() != null) {
            uris.add(data.getData());
        }
        toast("Import laeuft...");
        new Thread(() -> {
            int added = 0;
            for (Uri uri : uris) {
                try {
                    added += processImportedImage(uri);
                } catch (Exception ignored) {
                }
            }
            int finalAdded = added;
            runOnUiThread(() -> {
                saveState();
                toast(finalAdded + " Bilder/Crops importiert");
                render();
            });
        }).start();
    }

    private int processImportedImage(Uri uri) throws IOException {
        byte[] originalBytes;
        try (InputStream input = getContentResolver().openInputStream(uri)) {
            if (input == null) {
                return 0;
            }
            originalBytes = readBytes(input);
        }
        Bitmap bitmap = decodeScaled(originalBytes, 2200);
        if (bitmap == null) {
            return 0;
        }
        String displayName = displayName(uri);
        int added = 0;
        byte[] jpeg = bitmapToJpeg(bitmap, 90);
        ImageItem original = saveImportedJpeg(jpeg, displayName, bitmap.getWidth(), bitmap.getHeight());
        if (original != null) {
            added += 1;
        }

        ArrayList<CloudCropper.Candidate> candidates = CloudCropper.candidates(bitmap, 8);
        int cropIndex = 1;
        for (CloudCropper.Candidate crop : candidates) {
            Bitmap cropBitmap = Bitmap.createBitmap(bitmap, crop.x, crop.y, crop.size, crop.size);
            byte[] cropBytes = bitmapToJpeg(cropBitmap, 90);
            ImageItem cropItem = saveImportedJpeg(cropBytes, displayName + "_crop" + cropIndex, crop.size, crop.size);
            if (cropItem != null) {
                added += 1;
            }
            cropBitmap.recycle();
            cropIndex += 1;
        }
        bitmap.recycle();
        return added;
    }

    private ImageItem saveImportedJpeg(byte[] bytes, String originalName, int width, int height) throws IOException {
        String imageId = hash16(bytes);
        if (hasImageId(imageId)) {
            return null;
        }
        File dir = new File(getFilesDir(), "imported");
        if (!dir.exists() && !dir.mkdirs()) {
            throw new IOException("Unable to create import directory");
        }
        File file = new File(dir, imageId + ".jpg");
        try (FileOutputStream output = new FileOutputStream(file)) {
            output.write(bytes);
        }
        ImageItem item = new ImageItem();
        item.imageId = imageId;
        item.filePath = file.getAbsolutePath();
        item.originalPath = originalName;
        item.width = width;
        item.height = height;
        item.imported = true;
        importedImages.add(item);
        return item;
    }

    private void deleteImportedImages() {
        for (ImageItem item : importedImages) {
            if (item.filePath != null) {
                new File(item.filePath).delete();
            }
        }
        importedImages.clear();
        JSONArray keptRatings = new JSONArray();
        Set<String> allIds = new HashSet<>();
        for (ImageItem item : packagedImages) {
            allIds.add(item.imageId);
        }
        for (int i = 0; i < ratings.length(); i++) {
            JSONObject row = ratings.optJSONObject(i);
            if (row != null && allIds.contains(row.optString("image_id"))) {
                keptRatings.put(row);
            }
        }
        ratings = keptRatings;
    }

    private void exportLabelsByMail() {
        persistRater();
        try {
            File exportFile = createExportFile();
            Uri uri = Uri.parse("content://" + getPackageName() + ".exportfiles/" + exportFile.getName());
            Intent intent = new Intent(Intent.ACTION_SEND);
            intent.setType("application/json");
            if (!recipientEmail.trim().isEmpty()) {
                intent.putExtra(Intent.EXTRA_EMAIL, new String[] {recipientEmail.trim()});
            }
            intent.putExtra(Intent.EXTRA_SUBJECT, "Cloud labels von " + raterEdit.getText().toString().trim());
            intent.putExtra(Intent.EXTRA_TEXT, "Hier sind meine Cloud-Labels als JSON-Anhang.");
            intent.putExtra(Intent.EXTRA_STREAM, uri);
            intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
            startActivity(Intent.createChooser(intent, "Labels per Mail senden"));
        } catch (Exception error) {
            toast("Export fehlgeschlagen: " + error.getMessage());
        }
    }

    private File createExportFile() throws IOException, JSONException {
        JSONObject bundle = new JSONObject();
        bundle.put("version", 2);
        bundle.put("package_name", packageName);
        bundle.put("rater_id", raterEdit.getText().toString().trim());
        bundle.put("session_id", sessionId());
        bundle.put("exported_at", isoNow());
        bundle.put("ratings", new JSONArray(ratings.toString()));
        bundle.put("pairwise", new JSONArray(pairwise.toString()));
        JSONArray imports = new JSONArray();
        for (ImageItem item : importedImages) {
            File file = new File(item.filePath);
            if (!file.exists()) {
                continue;
            }
            JSONObject row = new JSONObject();
            row.put("image_id", item.imageId);
            row.put("name", item.originalPath);
            row.put("data_url", "data:image/jpeg;base64," + Base64.encodeToString(readBytes(new FileInputStream(file)), Base64.NO_WRAP));
            imports.put(row);
        }
        bundle.put("imported_images", imports);

        File dir = new File(getCacheDir(), "exports");
        if (!dir.exists() && !dir.mkdirs()) {
            throw new IOException("Unable to create export directory");
        }
        String safeRater = safeName(raterEdit.getText().toString().trim());
        File file = new File(dir, safeName(packageName) + "_" + safeRater + "_labels.json");
        try (OutputStreamWriter writer = new OutputStreamWriter(new FileOutputStream(file), StandardCharsets.UTF_8)) {
            writer.write(bundle.toString(2));
        }
        return file;
    }

    private ArrayList<ImageItem> allImages() {
        ArrayList<ImageItem> images = new ArrayList<>(packagedImages.size() + importedImages.size());
        images.addAll(packagedImages);
        images.addAll(importedImages);
        return images;
    }

    private boolean hasImageId(String imageId) {
        for (ImageItem item : packagedImages) {
            if (item.imageId.equals(imageId)) {
                return true;
            }
        }
        for (ImageItem item : importedImages) {
            if (item.imageId.equals(imageId)) {
                return true;
            }
        }
        return false;
    }

    private Set<String> ratedIds() {
        HashSet<String> ids = new HashSet<>();
        for (int i = 0; i < ratings.length(); i++) {
            JSONObject row = ratings.optJSONObject(i);
            if (row != null) {
                ids.add(row.optString("image_id"));
            }
        }
        return ids;
    }

    private double existingRatingScore(String imageId, double fallback) {
        for (int i = ratings.length() - 1; i >= 0; i--) {
            JSONObject row = ratings.optJSONObject(i);
            if (row != null && imageId.equals(row.optString("image_id"))) {
                return row.optDouble("score", fallback);
            }
        }
        return fallback;
    }

    private String existingRatingNote(String imageId) {
        for (int i = ratings.length() - 1; i >= 0; i--) {
            JSONObject row = ratings.optJSONObject(i);
            if (row != null && imageId.equals(row.optString("image_id"))) {
                return row.optString("note", "");
            }
        }
        return "";
    }

    private Bitmap loadBitmap(ImageItem item, int maxEdge) {
        try {
            if (item.imported) {
                return decodeScaled(readBytes(new FileInputStream(item.filePath)), maxEdge);
            }
            try (InputStream input = openPackagedAsset(item.assetPath)) {
                return decodeScaled(readBytes(input), maxEdge);
            }
        } catch (Exception error) {
            Bitmap fallback = Bitmap.createBitmap(32, 32, Bitmap.Config.ARGB_8888);
            fallback.eraseColor(PANEL_2);
            return fallback;
        }
    }

    private InputStream openPackagedAsset(String path) throws IOException {
        try {
            return getAssets().open(path);
        } catch (IOException first) {
            if (path.contains("/")) {
                return getAssets().open(path.replace("/", "\\"));
            }
            throw first;
        }
    }

    private Bitmap decodeScaled(byte[] bytes, int maxEdge) {
        BitmapFactory.Options bounds = new BitmapFactory.Options();
        bounds.inJustDecodeBounds = true;
        BitmapFactory.decodeByteArray(bytes, 0, bytes.length, bounds);
        int sample = 1;
        while (bounds.outWidth / sample > maxEdge || bounds.outHeight / sample > maxEdge) {
            sample *= 2;
        }
        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inSampleSize = sample;
        options.inPreferredConfig = Bitmap.Config.ARGB_8888;
        Bitmap decoded = BitmapFactory.decodeByteArray(bytes, 0, bytes.length, options);
        if (decoded == null) {
            return null;
        }
        int width = decoded.getWidth();
        int height = decoded.getHeight();
        int max = Math.max(width, height);
        if (max <= maxEdge) {
            return decoded;
        }
        double scale = maxEdge / (double) max;
        Bitmap resized = Bitmap.createScaledBitmap(decoded, Math.max(1, (int) Math.round(width * scale)), Math.max(1, (int) Math.round(height * scale)), true);
        decoded.recycle();
        return resized;
    }

    private byte[] bitmapToJpeg(Bitmap bitmap, int quality) {
        ByteArrayOutputStream output = new ByteArrayOutputStream();
        bitmap.compress(Bitmap.CompressFormat.JPEG, quality, output);
        return output.toByteArray();
    }

    private String readText(InputStream input) throws IOException {
        return new String(readBytes(input), StandardCharsets.UTF_8);
    }

    private byte[] readBytes(InputStream input) throws IOException {
        try (InputStream source = input; ByteArrayOutputStream output = new ByteArrayOutputStream()) {
            byte[] buffer = new byte[8192];
            int count;
            while ((count = source.read(buffer)) != -1) {
                output.write(buffer, 0, count);
            }
            return output.toByteArray();
        }
    }

    private String hash16(byte[] bytes) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            byte[] hashed = digest.digest(bytes);
            StringBuilder builder = new StringBuilder();
            for (int i = 0; i < 8; i++) {
                builder.append(String.format(Locale.US, "%02x", hashed[i] & 0xff));
            }
            return builder.toString();
        } catch (NoSuchAlgorithmException error) {
            return Long.toHexString(System.currentTimeMillis());
        }
    }

    private String displayName(Uri uri) {
        try (Cursor cursor = getContentResolver().query(uri, new String[] {OpenableColumns.DISPLAY_NAME}, null, null, null)) {
            if (cursor != null && cursor.moveToFirst()) {
                int index = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME);
                if (index >= 0) {
                    return cursor.getString(index);
                }
            }
        } catch (Exception ignored) {
        }
        return "imported_image";
    }

    private void persistRater() {
        String rater = raterEdit.getText().toString().trim();
        if (rater.isEmpty()) {
            rater = "friend";
            raterEdit.setText(rater);
        }
        getSharedPreferences("cloud_labeler", MODE_PRIVATE).edit().putString("rater_id", rater).apply();
    }

    private String sessionId() {
        return new SimpleDateFormat("yyyyMMdd", Locale.US).format(new Date()) + "_mobile_session";
    }

    private String isoNow() {
        return new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSSZ", Locale.US).format(new Date());
    }

    private String formatScore(double value) {
        return String.format(Locale.US, "%.1f", value);
    }

    private String safeName(String value) {
        StringBuilder builder = new StringBuilder();
        for (int i = 0; i < value.length(); i++) {
            char c = value.charAt(i);
            if (Character.isLetterOrDigit(c) || c == '-' || c == '_') {
                builder.append(c);
            } else {
                builder.append('_');
            }
        }
        if (builder.length() == 0) {
            return "friend";
        }
        return builder.toString();
    }

    private TextView heading(String text) {
        TextView view = new TextView(this);
        view.setText(text);
        view.setTextColor(FG);
        view.setTextSize(22);
        view.setTypeface(android.graphics.Typeface.DEFAULT_BOLD);
        return view;
    }

    private TextView body(String text) {
        TextView view = new TextView(this);
        view.setText(text);
        view.setTextColor(MUTED);
        view.setTextSize(15);
        view.setLineSpacing(dp(2), 1.0f);
        LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        params.topMargin = dp(10);
        view.setLayoutParams(params);
        return view;
    }

    private TextView emptyText(String text) {
        TextView view = new TextView(this);
        view.setText(text);
        view.setTextColor(MUTED);
        view.setTextSize(16);
        view.setGravity(Gravity.CENTER);
        return view;
    }

    private Button button(String text, boolean primary) {
        Button button = new Button(this);
        button.setAllCaps(false);
        button.setText(text);
        button.setTextColor(primary ? Color.WHITE : FG);
        button.setTextSize(13);
        button.setTypeface(android.graphics.Typeface.DEFAULT_BOLD);
        button.setBackground(panelDrawable(primary ? ACCENT : PANEL_2, dp(8), primary ? ACCENT : 0xFF313B48));
        return button;
    }

    private GradientDrawable panelDrawable(int color, int radius, int stroke) {
        GradientDrawable drawable = new GradientDrawable();
        drawable.setColor(color);
        drawable.setCornerRadius(radius);
        if (stroke != 0) {
            drawable.setStroke(dp(1), stroke);
        }
        return drawable;
    }

    private void toast(String message) {
        Toast.makeText(this, message, Toast.LENGTH_SHORT).show();
    }

    private int dp(int value) {
        return Math.round(value * getResources().getDisplayMetrics().density);
    }

    private int systemBarHeight(String name) {
        int id = getResources().getIdentifier(name, "dimen", "android");
        if (id <= 0) {
            return 0;
        }
        return getResources().getDimensionPixelSize(id);
    }

    private static class ImageItem {
        String imageId;
        String assetPath;
        String filePath;
        String originalPath;
        boolean imported;
        int width;
        int height;
    }

}
