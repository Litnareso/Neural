package org.pytorch.LSDnet;
import static android.graphics.Color.rgb;

import android.Manifest;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.media.MediaMetadataRetriever;
import android.net.Uri;
import android.os.Bundle;
import android.os.SystemClock;
import android.view.View;
import android.widget.TextView;
import android.widget.ImageView;
import android.widget.ImageButton;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import org.pytorch.IValue;
import org.pytorch.LiteModuleLoader;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;


public class MainActivity extends AppCompatActivity implements Runnable {

    public static String assetFilePath(Context context, String assetName) throws IOException {
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        }
    }


    private int step = 200;
    private Bitmap floatArrayToBitmap(float [] floatArray, int width, int height) {

        // Create empty bitmap in ARGB format
        Bitmap bmp = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        int[] pixels = new int[width * height * 4];

        // mapping smallest value to 0 and largest value to 255
        float maxValue = -1;
        float minValue = 1;
        for (int i = 0; i < floatArray.length; i++) {
            maxValue = Math.max(floatArray[i], maxValue);
            minValue = Math.min(floatArray[i], minValue);
        }
        float delta = maxValue - minValue;
        // copy each value from float array to RGB channels
        for (int i = 0; i < width* height; i++) {
            int r = (int)((floatArray[i] - minValue) / delta * 255.);
            int g = (int)((floatArray[i+width*height] - minValue) / delta * 255.);
            int b =(int)((floatArray[i+2*width*height] - minValue) / delta * 255.);
            r = Math.min(Math.max(r, 0), 255);
            g = Math.min(Math.max(g, 0), 255);
            b = Math.min(Math.max(b, 0), 255);
            pixels[i] = rgb(r, g, b); // you might need to import for rgb()
        }
        bmp.setPixels(pixels, 0, width, 0, 0, width, height);

        return bmp;
    }
    private boolean to_wait = false;
    private TextView mTextView;
    private String mImagename = "dog.jpg";
    private ImageView mImageView;
    private Bitmap bitmap = null;
    private  int durationTo = 0;
    public static Module model = null;
    private double durationMs = 0;
    private Bitmap buf;
    Uri mVideoUri;
    private MediaMetadataRetriever mmr = null;
    private Module model_enhancer = null;
    private int state = 0;

    Bitmap Model_forward(Bitmap buf) {
        final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(buf,
                TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB);
        final float[] outimgTensor = model_enhancer.forward(IValue.from(inputTensor)).toTensor().getDataAsFloatArray();
        Bitmap tmp = floatArrayToBitmap(outimgTensor, buf.getWidth(),buf.getHeight());
        return tmp;
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.READ_EXTERNAL_STORAGE}, 1);
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, 1);
        }

        setContentView(R.layout.activity_main);
        if (model_enhancer == null) {
            try {
                model_enhancer = LiteModuleLoader.load(MainActivity.assetFilePath(this.getApplicationContext(), "model.ptl"));
            } catch (IOException e) {
            }
        }
        if (buf == null) {
            try {
                buf = BitmapFactory.decodeStream(getAssets().open(mImagename));

            } catch (IOException e) {

            }
        }
        mTextView = findViewById(R.id.textViewR);
        mImageView = findViewById(R.id.imageView);
        Bitmap tmp = Model_forward(buf);
        mImageView.setImageBitmap(tmp);

        final ImageButton buttonSelect = findViewById(R.id.gallerybutton);
        buttonSelect.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                Intent pickIntent = new Intent(Intent.ACTION_PICK, android.provider.MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                pickIntent.setType("video/*");
                startActivityForResult(pickIntent, 1);
            }
        });

        final ImageButton buttonCamera = findViewById(R.id.camerabutton);
        buttonCamera.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                final Intent intent = new Intent(MainActivity.this, LiveVideoClassificationActivity.class);
                startActivity(intent);
            }
        });
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode == RESULT_OK && data != null) {
            Uri selectedMediaUri = data.getData();
            if (selectedMediaUri.toString().contains("video")) {
                mVideoUri = selectedMediaUri;
            }
            state = 1;
        }

        int a = 1;


                mmr = new MediaMetadataRetriever();
                mmr.setDataSource(this.getApplicationContext(), mVideoUri);
                String stringDuration = mmr.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION);
                durationMs = Double.parseDouble(stringDuration);

                // for each second of the video, make inference to get the class label



                 Thread thread = new Thread(MainActivity.this);
                 thread.start();




    }
    @Override
    public void run() {

        durationTo = (int) Math.ceil(durationMs / step);
        long timeMs =0;
        int i = 0;
        while (timeMs < durationMs) {
            long timeUs =  timeMs * 1000;
            bitmap = mmr.getFrameAtTime(timeUs, MediaMetadataRetriever.OPTION_CLOSEST);

            bitmap = Bitmap.createScaledBitmap(bitmap, (int)(256), (int)(256), true);
            final long startTime = SystemClock.elapsedRealtime();
            bitmap = Model_forward(bitmap);
            final Bitmap transferredBitmap = Bitmap.createScaledBitmap(bitmap, 1024, 1024, true);
            final long inferenceTime = SystemClock.elapsedRealtime() - startTime;
            timeMs += inferenceTime;
            final String str_1 = String.format("Time: %dms", inferenceTime);
            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    to_wait = false;
                    mImageView.setImageBitmap(transferredBitmap);
                    mTextView.setText(str_1);
                }
            });

            i++;

        }



    }
}
