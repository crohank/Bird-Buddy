package com.example.objectprediction;

import android.annotation.SuppressLint;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.ColorSpace;
import android.media.ThumbnailUtils;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import com.example.objectprediction.ml.Linear;
import com.example.objectprediction.ml.Linear1;
import com.example.objectprediction.ml.Linear3;
import com.example.objectprediction.ml.Mobile;
import com.example.objectprediction.ml.MobilenetV110224Quant;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class MainActivity extends AppCompatActivity {
    Button capture,upload,predict;
    TextView result;
    ImageView image;

    Bitmap bitmap;
    private Bitmap bitmap_data;
    int imageSize = 150;
    int cam = 0;
    @SuppressLint("SetTextI18n")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        String[] labels = new String[526];
        int cnt=0;
        try {
            BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(getAssets().open("labels-birds1.txt")));
            String line = bufferedReader.readLine();
            while(line!= null){
                labels[cnt] = line;
                cnt++;
                line = bufferedReader.readLine();
            }
        }
        catch (IOException e){
            e.printStackTrace();
        }

        capture = findViewById(R.id.capture);
        upload = findViewById(R.id.upload);
        predict = findViewById(R.id.predict);

        result = findViewById(R.id.result);
        image = findViewById(R.id.imageView);

        upload.setOnClickListener(v -> {
            result.setText("");
            Intent intent = new Intent();
            intent.setAction(Intent.ACTION_GET_CONTENT);
            intent.setType("image/*");
            startActivityForResult(intent,10);
        });
        capture.setOnClickListener(v -> {
            result.setText("");
            Intent camera_intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
            startActivityForResult(camera_intent, 11);
        });
        predict.setOnClickListener(v -> {
            try {
                Mobile model = Mobile.newInstance(getApplicationContext());
                TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 150, 150, 3}, DataType.FLOAT32);
                ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3);
                byteBuffer.order(ByteOrder.nativeOrder());

                int[] intValues = new int[imageSize * imageSize];

                if (cam == 1) {

                    int capturedImageWidth = bitmap.getWidth();
                    int capturedImageHeight = bitmap.getHeight();
                    int x = 0;
                    int width = Math.min(imageSize, capturedImageWidth);

                    bitmap.getPixels(intValues, 0, width, x, 0, width, Math.min(imageSize, capturedImageHeight));
                } else {

                    bitmap.getPixels(intValues, 0, imageSize, 0, 0, Math.min(imageSize, image.getWidth()), Math.min(imageSize, image.getHeight()));
                }
                int pixel = 0;

                for (int i = 0; i < imageSize; i++) {
                    for (int j = 0; j < imageSize; j++) {
                        int val = intValues[pixel++]; // RGB
                        byteBuffer.putFloat(((val >> 16) & 0xFF) * (1.f / 255));
                        byteBuffer.putFloat(((val >> 8) & 0xFF) * (1.f / 255));
                        byteBuffer.putFloat((val & 0xFF) * (1.f / 255));
                    }
                }

                inputFeature0.loadBuffer(byteBuffer);


                Mobile.Outputs outputs = model.process(inputFeature0);
                TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();
                int maxPos = 0;
                float[] confidences = outputFeature0.getFloatArray();
                float maxConfidence = 0;
                for (int i = 0; i < confidences.length; i++) {
                    if (confidences[i] > maxConfidence) {
                        maxConfidence = confidences[i];
                        maxPos = i;
                    }
                }

                //result.setText("The bird is called "+labels[maxPos]);
                if(labels[maxPos].equals("Not Bird")){
                    result.setText("It is not a bird");
                }
                else{
                    result.setText("The bird is called\n \n" + labels[maxPos]);
                }
                model.close();
            } catch (IOException e) {
                // TODO Handle the exception
            }

        });
    }

    int getMax(float[] arr){
        int max = 0;
        for(int i=0;i<arr.length;i++){
            if(arr[i]>arr[max]){
                max = i;
            }
        }
        return max;
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == 11) {
            cam=1;
            assert data != null;
            bitmap = (Bitmap) data.getExtras().get("data");
            image.setImageBitmap(bitmap);

        }
        if (requestCode == 10) {
                if (data != null) {
                    Uri uri = data.getData();
                    try {
                        bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), uri);
                        image.setImageBitmap(bitmap);
                    } catch (IOException e) {
                        throw new RuntimeException(e);
                    }
                }
            }
    }
}