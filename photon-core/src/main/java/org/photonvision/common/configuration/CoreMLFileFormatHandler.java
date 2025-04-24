package org.photonvision.common.configuration;

import io.javalin.http.UploadedFile;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Optional;
import org.photonvision.common.logging.LogGroup;
import org.photonvision.common.logging.Logger;
import org.photonvision.vision.objects.CoreMLFileModel;
import org.photonvision.vision.objects.Model;

public class CoreMLFileFormatHandler implements ModelFormatHandler {

    private static final Logger logger = new Logger(CoreMLFileFormatHandler.class, LogGroup.Config);
    private static final String BACKEND_NAME = "COREML_FILE";
    private static final String PRIMARY_EXTENSION = ".mlmodel";
    private static final String UPLOAD_EXTENSION = ".mlmodel";
    private static final Class<? extends Model> MODEL_CLASS = CoreMLFileModel.class;

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
        return modelFileName.endsWith(UPLOAD_EXTENSION) && labelsFileName.endsWith("-labels.txt");
    }

    @Override
    public Optional<String> validateUpload(UploadedFile modelFile, UploadedFile labelsFile) {
        String modelExtension = modelFile.extension().toLowerCase();
        String labelsExtension = labelsFile.extension().toLowerCase();

        if (!modelExtension.equals(UPLOAD_EXTENSION)) {
            return Optional.of("Invalid model file type. Expected '" + UPLOAD_EXTENSION + "' but got '" + modelExtension + "'");
        }
        if (!labelsExtension.equals(".txt")) {
            return Optional.of("Invalid labels file type. Expected '.txt' but got '" + labelsExtension + "'");
        }
        return Optional.empty();
    }

    @Override
    public void verifyNames(String modelFileName, String labelsFileName) throws IllegalArgumentException {
        CoreMLFileModel.verifyNames(modelFileName, labelsFileName);
    }

    @Override
    public Model loadFromPath(Path path, File modelsDirectory) throws IOException, IllegalArgumentException {
        String labelsFileName = deriveLabelsName(path.getFileName().toString(), PRIMARY_EXTENSION);
        Path labelsPath = modelsDirectory.toPath().resolve(labelsFileName);
        // CoreMLFileModel constructor takes File and String path for labels
        return new CoreMLFileModel(path.toFile(), labelsPath.toString());
    }

    @Override
    public void saveUploadedFiles(UploadedFile modelFile, UploadedFile labelsFile, File modelsDirectory) throws IOException {
        Path labelsDestPath = modelsDirectory.toPath().resolve(labelsFile.filename());
        Path modelDestPath = modelsDirectory.toPath().resolve(modelFile.filename());

        logger.info("Saving CoreML files to: " + modelsDirectory);
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
            try { Files.deleteIfExists(labelsDestPath); } catch (IOException ignored) {}
            throw new IOException("Failed to save CoreML model file: " + modelDestPath, e);
        }
    }

    // Helper method
    private String deriveLabelsName(String modelFileName, String modelExtension)
            throws IllegalArgumentException {
        if (!modelFileName.endsWith(modelExtension)) {
            throw new IllegalArgumentException(
                    "Model filename '" + modelFileName + "' does not end with expected extension '"
                            + modelExtension + "'");
        }
        String baseName = modelFileName.substring(0, modelFileName.length() - modelExtension.length());
        return baseName + "-labels.txt";
    }
} 