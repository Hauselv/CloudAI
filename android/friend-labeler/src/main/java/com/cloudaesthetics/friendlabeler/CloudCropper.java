package com.cloudaesthetics.friendlabeler;

import android.graphics.Bitmap;
import android.graphics.Color;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;

final class CloudCropper {
    private CloudCropper() {
    }

    static ArrayList<Candidate> candidates(Bitmap bitmap, int limit) {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        int shortEdge = Math.min(width, height);
        int[] pixels = new int[width * height];
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height);
        double[] scales = new double[] {0.35, 0.5, 0.7, 0.9, 1.0};
        ArrayList<Candidate> candidates = new ArrayList<>();
        for (double scale : scales) {
            int size = Math.max(384, (int) Math.round(shortEdge * scale));
            if (size > width || size > height) {
                continue;
            }
            int step = Math.max(1, (int) Math.round(size * 0.5));
            ArrayList<Integer> xs = positions(width, size, step);
            ArrayList<Integer> ys = positions(height, size, step);
            for (int y : ys) {
                for (int x : xs) {
                    double[] stats = cropStats(pixels, width, x, y, size);
                    if (stats[0] >= 0.72 && stats[1] >= 0.08) {
                        Candidate candidate = new Candidate();
                        candidate.x = x;
                        candidate.y = y;
                        candidate.size = size;
                        candidate.sky = stats[0];
                        candidate.cloud = stats[1];
                        candidates.add(candidate);
                    }
                }
            }
        }
        Collections.sort(candidates, new Comparator<Candidate>() {
            @Override
            public int compare(Candidate a, Candidate b) {
                double scoreA = a.sky + a.cloud * 0.5 + a.size / 10000.0;
                double scoreB = b.sky + b.cloud * 0.5 + b.size / 10000.0;
                return Double.compare(scoreB, scoreA);
            }
        });
        if (candidates.size() > limit) {
            return new ArrayList<>(candidates.subList(0, limit));
        }
        return candidates;
    }

    private static ArrayList<Integer> positions(int length, int size, int step) {
        ArrayList<Integer> values = new ArrayList<>();
        int end = length - size;
        for (int value = 0; value <= end; value += step) {
            values.add(value);
        }
        if (values.isEmpty() || values.get(values.size() - 1) != end) {
            values.add(end);
        }
        return values;
    }

    private static double[] cropStats(int[] pixels, int width, int x, int y, int size) {
        int step = Math.max(1, size / 64);
        int sky = 0;
        int cloud = 0;
        int total = 0;
        float[] hsv = new float[3];
        for (int yy = y; yy < y + size; yy += step) {
            for (int xx = x; xx < x + size; xx += step) {
                int color = pixels[yy * width + xx];
                Color.RGBToHSV(Color.red(color), Color.green(color), Color.blue(color), hsv);
                double hue = hsv[0] / 2.0;
                double saturation = hsv[1] * 255.0;
                double value = hsv[2] * 255.0;
                boolean blueSky = hue >= 85 && hue <= 135 && saturation >= 20 && value >= 70;
                boolean brightCloud = saturation <= 95 && value >= 115;
                boolean hazySky = saturation <= 70 && value >= 145;
                if (blueSky || brightCloud || hazySky) {
                    sky += 1;
                }
                if (saturation <= 85 && value >= 125) {
                    cloud += 1;
                }
                total += 1;
            }
        }
        if (total == 0) {
            return new double[] {0.0, 0.0};
        }
        return new double[] {sky / (double) total, cloud / (double) total};
    }

    static final class Candidate {
        int x;
        int y;
        int size;
        double sky;
        double cloud;
    }
}
