package org.photonvision.vision.objects;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.opencv.core.Size;
import org.photonvision.common.logging.LogGroup;
import org.photonvision.common.logging.Logger;
import org.photonvision.coreml.CoreMLJNI;
import org.photonvision.jni.CoreMLObjectDetector;

/** 
 * Represents a CoreML model that can be used for object detection.
 * Supports both *.mlmodel files and *.mlpackage directories.
 */
public class CoreMLModel implements Model {
    private static final Logger logger = new Logger(CoreMLModel.class, LogGroup.Config);

    /** The file or directory containing the model. */
    public final Path modelPath;
    
    /** Flag indicating if this is a .mlpackage directory model */
    public final boolean isPackage;

    /** The labels that the model can detect. */
    public final List<String> labels;

    /** The version of the model. */
    public final CoreMLJNI.ModelVersion version;

    /** The input size of the model. */
    public final Size inputSize;

    /**
     * Creates a new CoreML model.
     * Assumes parameters have been validated and parsed by the format handler.
     *
     * @param modelPath The path to the model file or directory.
     * @param isPackage True if this is a .mlpackage directory model, false for .mlmodel file.
     * @param labels The labels that the model can detect.
     * @param version The version of the model.
     * @param inputSize The input size required by the model.
     * @throws IllegalArgumentException If the model path is invalid.
     */
    public CoreMLModel(Path modelPath, boolean isPackage, List<String> labels, 
                       CoreMLJNI.ModelVersion version, Size inputSize) 
            throws IllegalArgumentException {
        this.modelPath = modelPath;
        this.isPackage = isPackage;
        this.labels = labels;
        this.version = version;
        this.inputSize = inputSize;

        // Basic validation of the path
        if (isPackage) {
            if (!Files.isDirectory(modelPath)) {
                throw new IllegalArgumentException("Package path must be an existing directory: " + modelPath);
            }
        } else {
            if (!Files.isRegularFile(modelPath)) {
                throw new IllegalArgumentException("Model file path must be an existing file: " + modelPath);
            }
        }

        logger.info("CoreMLModel created for: " + modelPath.getFileName() + 
                    ", Type: " + (isPackage ? "package" : "file") + 
                    ", Version: " + version + ", Size: " + inputSize);
    }

    @Override
    public String getName() {
        return modelPath.getFileName().toString();
    }

    @Override
    public ObjectDetector load() {
        // Check path exists before loading
        if (isPackage ? !Files.isDirectory(modelPath) : !Files.isRegularFile(modelPath)) {
            logger.error("Model " + (isPackage ? "directory" : "file") + 
                         " does not exist when trying to load: " + modelPath);
            return null;
        }
        
        try {
            return new CoreMLObjectDetector(this, inputSize);
        } catch (Exception e) {
            logger.error("Failed to load CoreML model: " + modelPath.getFileName(), e);
            return null;
        }
    }
} 