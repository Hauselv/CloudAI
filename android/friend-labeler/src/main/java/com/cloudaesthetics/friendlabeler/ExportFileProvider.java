package com.cloudaesthetics.friendlabeler;

import android.content.ContentProvider;
import android.content.ContentValues;
import android.database.Cursor;
import android.database.MatrixCursor;
import android.net.Uri;
import android.os.ParcelFileDescriptor;
import android.provider.OpenableColumns;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;

public class ExportFileProvider extends ContentProvider {
    @Override
    public boolean onCreate() {
        return true;
    }

    @Override
    public String getType(Uri uri) {
        return "application/json";
    }

    @Override
    public ParcelFileDescriptor openFile(Uri uri, String mode) throws FileNotFoundException {
        return ParcelFileDescriptor.open(resolveExportFile(uri), ParcelFileDescriptor.MODE_READ_ONLY);
    }

    @Override
    public Cursor query(Uri uri, String[] projection, String selection, String[] selectionArgs, String sortOrder) {
        File file = resolveExportFile(uri);
        MatrixCursor cursor = new MatrixCursor(new String[] {OpenableColumns.DISPLAY_NAME, OpenableColumns.SIZE});
        cursor.addRow(new Object[] {file.getName(), file.length()});
        return cursor;
    }

    @Override
    public Uri insert(Uri uri, ContentValues values) {
        throw new UnsupportedOperationException("Read-only provider");
    }

    @Override
    public int delete(Uri uri, String selection, String[] selectionArgs) {
        throw new UnsupportedOperationException("Read-only provider");
    }

    @Override
    public int update(Uri uri, ContentValues values, String selection, String[] selectionArgs) {
        throw new UnsupportedOperationException("Read-only provider");
    }

    private File resolveExportFile(Uri uri) {
        if (getContext() == null) {
            throw new IllegalStateException("Provider has no context");
        }
        String name = uri.getLastPathSegment();
        if (name == null || name.contains("/") || name.contains("\\")) {
            throw new IllegalArgumentException("Invalid export file name");
        }
        File root = new File(getContext().getCacheDir(), "exports");
        File file = new File(root, name);
        try {
            String rootPath = root.getCanonicalPath();
            String filePath = file.getCanonicalPath();
            if (!filePath.startsWith(rootPath + File.separator)) {
                throw new IllegalArgumentException("Export file outside cache root");
            }
        } catch (IOException error) {
            throw new IllegalArgumentException("Unable to resolve export file", error);
        }
        if (!file.exists()) {
            throw new IllegalArgumentException("Export file does not exist");
        }
        return file;
    }
}
