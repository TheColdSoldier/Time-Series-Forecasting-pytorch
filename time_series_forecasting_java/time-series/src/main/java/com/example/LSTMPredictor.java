import org.tensorflow.keras.models.*;
import org.tensorflow.keras.layers.*;
import org.tensorflow.keras.optimizers.*;
import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;
import org.tensorflow.ndarray.NdArrays;
import org.tensorflow.types.TFloat32;

public class LSTMPredictor {
    public static void main(String[] args) throws IOException {
        // Load and preprocess stock data
        List<Float> stockPrices = loadStockData("data/stock_prices.csv");
        float[][] normalizedData = normalizeData(stockPrices);
        
        int seqLength = 10;
        float[][] X = new float[normalizedData.length - seqLength][seqLength];
        float[] y = new float[normalizedData.length - seqLength];
        
        for (int i = 0; i < normalizedData.length - seqLength; i++) {
            System.arraycopy(normalizedData[i], 0, X[i], 0, seqLength);
            y[i] = normalizedData[i + seqLength][0];
        }
        
        // Convert to TensorFlow tensors
        TFloat32 X_train = TFloat32.tensorOf(NdArrays.ofFloats(X));
        TFloat32 y_train = TFloat32.tensorOf(NdArrays.vectorOf(y));
        
        // Build LSTM model
        Sequential model = new Sequential();
        model.add(new LSTM(64, InputShape.of(seqLength, 1)));
        model.add(new Dense(1));
        
        model.compile(new Adam(0.001f), "mean_squared_error");
        
        // Train the model
        model.fit(X_train, y_train, 50, 8, true);
        
        System.out.println("Model trained successfully!");
    }
    
    private static List<Float> loadStockData(String filePath) throws IOException {
        return Files.lines(Paths.get(filePath))
                .skip(1) 
                .map(line -> Float.parseFloat(line.split(",")[1])) // Assume 'Close' price is at index 1
                .collect(Collectors.toList());
    }
    
    private static float[][] normalizeData(List<Float> data) {
        float min = Collections.min(data);
        float max = Collections.max(data);
        float[][] normalized = new float[data.size()][1];
        for (int i = 0; i < data.size(); i++) {
            normalized[i][0] = (data.get(i) - min) / (max - min);
        }
        return normalized;
    }
}
