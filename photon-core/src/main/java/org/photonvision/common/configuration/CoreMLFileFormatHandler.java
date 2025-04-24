package org.photonvision.common.configuration;

import io.javalin.http.UploadedFile;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Optional;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.opencv.core.Size;
import org.photonvision.common.logging.LogGroup;
import org.photonvision.common.logging.Logger;
import org.photonvision.coreml.CoreMLJNI;
import org.photonvision.vision.objects.CoreMLModel;
import org.photonvision.vision.objects.Model;

public class CoreMLFileFormatHandler implements ModelFormatHandler {

    private static final Logger logger = new Logger(CoreMLFileFormatHandler.class, LogGroup.Config);
    private static final String BACKEND_NAME = "COREML_FILE";
    private static final String PRIMARY_EXTENSION = ".mlmodel";
    private static final String UPLOAD_EXTENSION = ".mlmodel";
    private static final Class<? extends Model> MODEL_CLASS = CoreMLModel.class;

    // Naming convention patterns
    private static final Pattern modelPattern =
            Pattern.compile("^([a-zA-Z0-9._-]+)-(\\d+)-(\\d+)-(yolov(?:5|8|11)[nsmlx]*)\\.mlmodel$");
    private static final Pattern labelsPattern =
            Pattern.compile("^([a-zA-Z0-9._-]+)-(\\d+)-(\\d+)-(yolov(?:5|8|11)[nsmlx]*)-labels\\.txt$");

    // Helper class to hold parsed name components
    private static class ParsedModelInfo {
        final String baseName;
        final int width;
        final int height;
        final String versionString;
        final CoreMLJNI.ModelVersion version;
        final Size inputSize;

        ParsedModelInfo(String baseName, int width, int height, String versionString) {
            this.baseName = baseName;
            this.width = width;
            this.height = height;
            this.versionString = versionString;
            this.version = parseVersionString(versionString); // Determine enum version
            this.inputSize = new Size(width, height);
        }
    }

    @Override
    public String getBackendName() {
        return BACKEND_NAME;
    }

    @Override
    public String getUploadAcceptType() {
        return UPLOAD_EXTENSION;
    }

    @Override
    public Class<? extends Model> getModelClass() {
        return MODEL_CLASS;
    }

    @Override
    public Info getInfo() {
        return new Info(getBackendName(), getUploadAcceptType());
    }

    @Override
    public boolean supportsPath(Path path) {
        if (path == null) return false;
        return Files.isRegularFile(path) && path.getFileName().toString().endsWith(PRIMARY_EXTENSION);
    }

    @Override
    public boolean supportsUpload(String modelFileName, String labelsFileName) {
        if (modelFileName == null || labelsFileName == null) return false;
        // Use verifyNames logic for consistency, or simple endsWith checks
        try {
            verifyNames(modelFileName, labelsFileName);
            return true;
        } catch (IllegalArgumentException e) {
            return false;
        }
    }

    @Override
    public Optional<String> validateUpload(UploadedFile modelFile, UploadedFile labelsFile) {
        // Basic extension check
        String modelExtension = getExtension(modelFile.filename()).toLowerCase();
        String labelsExtension = getExtension(labelsFile.filename()).toLowerCase();

        if (!modelExtension.equals(UPLOAD_EXTENSION)) {
            return Optional.of("Invalid model file type. Expected '" + UPLOAD_EXTENSION + "' but got '" + modelExtension + "'");
        }
        if (!labelsExtension.equals(".txt")) {
            return Optional.of("Invalid labels file type. Expected '.txt' but got '" + labelsExtension + "'");
        }

        // Deeper validation using naming convention
        try {
            verifyNames(modelFile.filename(), labelsFile.filename());
        } catch (IllegalArgumentException e) {
            return Optional.of("Invalid file naming convention: " + e.getMessage());
        }

        return Optional.empty();
    }

    @Override
    public void verifyNames(String modelFileName, String labelsFileName) throws IllegalArgumentException {
        // check null
        if (modelFileName == null || labelsFileName == null) {
            throw new IllegalArgumentException("Model name and labels name cannot be null");
        }

        Matcher modelMatcher = modelPattern.matcher(modelFileName);
        Matcher labelsMatcher = labelsPattern.matcher(labelsFileName);

        logger.debug("Verifying CoreML names - Model: " + modelFileName + ", Labels: " + labelsFileName);

        if (!modelMatcher.matches()) {
            throw new IllegalArgumentException(
                    "Model name '" + modelFileName + "' must follow the convention name-width-height-version" + PRIMARY_EXTENSION);
        }
        if (!labelsMatcher.matches()) {
            throw new IllegalArgumentException(
                    "Labels name '" + labelsFileName + "' must follow the convention name-width-height-version-labels.txt");
        }

        // Check if all captured groups match between the model and txt file names
        if (!modelMatcher.group(1).equals(labelsMatcher.group(1)) // baseName
                || !modelMatcher.group(2).equals(labelsMatcher.group(2)) // width
                || !modelMatcher.group(3).equals(labelsMatcher.group(3)) // height
                || !modelMatcher.group(4).equals(labelsMatcher.group(4))) { // versionString
            throw new IllegalArgumentException(
                "Model name ('" + modelFileName + ") and labels name ('" + labelsFileName + ") parts must match.");
        }

        // Additionally parse and check numeric parts
        try {
            parseModelName(modelFileName); // This implicitly validates numeric parts and version string
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException("Invalid components in model name: " + e.getMessage(), e);
        }
    }

    @Override
    public Model loadFromPath(Path path, File modelsDirectory) throws IOException, IllegalArgumentException {
        String modelFileName = path.getFileName().toString();
        logger.info("Loading CoreML model from path: " + path);

        // 1. Parse model name to get info (implicitly validates format)
        ParsedModelInfo parsedInfo;
        try {
            parsedInfo = parseModelName(modelFileName);
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException("Failed to parse model filename: " + modelFileName, e);
        }

        // 2. Determine and find the labels file
        String labelsFileName = deriveLabelsName(parsedInfo);
        Path labelsPath = modelsDirectory.toPath().resolve(labelsFileName);

        if (!Files.exists(labelsPath)) {
            throw new IOException("Could not find expected labels file: " + labelsPath);
        }
        if (!Files.isRegularFile(labelsPath)) {
            throw new IOException("Expected labels file is not a regular file: " + labelsPath);
        }

        // 3. Read labels
        List<String> labels;
        try {
            labels = Files.readAllLines(labelsPath);
            logger.debug("Successfully read labels from: " + labelsPath);
        } catch (IOException e) {
            throw new IOException("Failed to read labels file: " + labelsPath, e);
        }

        // 4. Create the CoreMLModel instance with parsed info
        try {
             // Updated to use the new unified CoreMLModel
             return new CoreMLModel(path, false, labels, parsedInfo.version, parsedInfo.inputSize);
        } catch (Exception e) {
             throw new IOException("Failed to instantiate CoreMLModel for " + modelFileName, e);
        }
    }

    @Override
    public void saveUploadedFiles(UploadedFile modelFile, UploadedFile labelsFile, File modelsDirectory) throws IOException {
        // Ensure target directory exists
        if (!modelsDirectory.exists()) {
            if (!modelsDirectory.mkdirs()) {
                throw new IOException("Failed to create models directory: " + modelsDirectory.getAbsolutePath());
            }
        }
        if (!modelsDirectory.isDirectory()) {
            throw new IOException("Models directory path is not a directory: " + modelsDirectory.getAbsolutePath());
        }

        // Verify names before saving
        try {
             verifyNames(modelFile.filename(), labelsFile.filename());
        } catch (IllegalArgumentException e) {
             throw new IOException("Uploaded file names are invalid: " + e.getMessage(), e);
        }

        Path labelsDestPath = modelsDirectory.toPath().resolve(labelsFile.filename());
        Path modelDestPath = modelsDirectory.toPath().resolve(modelFile.filename());

        logger.info("Saving CoreML files to: " + modelsDirectory.getAbsolutePath());
        logger.debug("Saving labels to: " + labelsDestPath);
        logger.debug("Saving model to: " + modelDestPath);

        // Save labels first
        try (InputStream in = labelsFile.content(); OutputStream out = Files.newOutputStream(labelsDestPath)) {
            in.transferTo(out);
        } catch (IOException e) {
            throw new IOException("Failed to save CoreML labels file: " + labelsDestPath, e);
        }

        // Save model file
        try (InputStream in = modelFile.content(); OutputStream out = Files.newOutputStream(modelDestPath)) {
            in.transferTo(out);
        } catch (IOException e) {
            // Clean up labels file if model save fails
            try {
                Files.deleteIfExists(labelsDestPath);
                logger.warn("Deleted labels file " + labelsDestPath + " due to model save failure.");
            } catch (IOException ignored) {
                logger.error("Failed to delete labels file " + labelsDestPath + " after model save failure.", ignored);
            }
            throw new IOException("Failed to save CoreML model file: " + modelDestPath, e);
        }
        logger.info("Successfully saved CoreML model " + modelFile.filename() + " and labels " + labelsFile.filename());
    }

    // --- Helper Methods --- //

    /**
     * Parse CoreML model file name and return parsed info.
     *
     * @param modelFileName the name of the model file (e.g., my_model-640-480-yolov8.mlmodel)
     * @throws IllegalArgumentException if the model name does not follow the naming convention
     * @return ParsedModelInfo containing components
     */
    private ParsedModelInfo parseModelName(String modelFileName) throws IllegalArgumentException {
        Matcher modelMatcher = modelPattern.matcher(modelFileName);

        if (!modelMatcher.matches()) {
            throw new IllegalArgumentException(
                    "Model name '" + modelFileName + "' must follow the convention name-width-height-version" + PRIMARY_EXTENSION);
        }

        try {
            String baseName = modelMatcher.group(1);
            int width = Integer.parseInt(modelMatcher.group(2));
            int height = Integer.parseInt(modelMatcher.group(3));
            String versionString = modelMatcher.group(4);

            if (width <= 0 || height <= 0) {
                throw new IllegalArgumentException("Width and height must be positive integers.");
            }

            // Validate version string via parseVersionString
            parseVersionString(versionString); // Throws if invalid

            return new ParsedModelInfo(baseName, width, height, versionString);
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException("Invalid width/height number in model name: " + modelFileName, e);
        } catch (IllegalArgumentException e) { // Catch version parsing errors
             throw new IllegalArgumentException("Invalid version string in model name: " + modelFileName + " -> " + e.getMessage(), e);
        }
    }

    /**
     * Determines the model version enum based on the version string from the filename.
     *
     * @param versionString The version part of the filename (e.g., "yolov5s", "yolov8n", "yolov11")
     * @return The corresponding CoreMLJNI.ModelVersion enum
     * @throws IllegalArgumentException if the version string is unknown
     */
    private static CoreMLJNI.ModelVersion parseVersionString(String versionString) throws IllegalArgumentException {
        // Normalize by checking the start of the string
        if (versionString.startsWith("yolov5")) {
            return CoreMLJNI.ModelVersion.YOLO_V5;
        } else if (versionString.startsWith("yolov8")) {
            return CoreMLJNI.ModelVersion.YOLO_V8;
        } else if (versionString.startsWith("yolov11")) {
            return CoreMLJNI.ModelVersion.YOLO_V11;
        } else {
            throw new IllegalArgumentException("Unknown model version string: " + versionString);
        }
    }

    /**
     * Derive the labels filename from parsed model info.
     */
    private String deriveLabelsName(ParsedModelInfo parsedInfo) {
        return parsedInfo.baseName + "-" + parsedInfo.width + "-" + parsedInfo.height + "-" + parsedInfo.versionString + "-labels.txt";
    }

    /** Helper to get file extension */
    private String getExtension(String filename) {
        int lastDot = filename.lastIndexOf('.');
        if (lastDot == -1) {
            return ""; // No extension
        }
        return filename.substring(lastDot);
    }
} 