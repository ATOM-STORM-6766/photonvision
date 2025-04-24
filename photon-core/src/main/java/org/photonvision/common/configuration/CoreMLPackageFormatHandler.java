package org.photonvision.common.configuration;

import io.javalin.http.UploadedFile;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Optional;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;
import org.photonvision.common.logging.LogGroup;
import org.photonvision.common.logging.Logger;
import org.photonvision.vision.objects.CoreMLPackageModel;
import org.photonvision.vision.objects.Model;
// Consider adding FileUtils if robust directory deletion on failure is needed
// import org.apache.commons.io.FileUtils;

public class CoreMLPackageFormatHandler implements ModelFormatHandler {

    private static final Logger logger = new Logger(CoreMLPackageFormatHandler.class, LogGroup.Config);
    private static final String BACKEND_NAME = "COREML_PACKAGE";
    private static final String PRIMARY_EXTENSION = ".mlpackage";
    private static final String UPLOAD_EXTENSION = ".zip";
    private static final Class<? extends Model> MODEL_CLASS = CoreMLPackageModel.class;

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
        // Check if it's a directory ending with the primary extension
        return Files.isDirectory(path) && path.getFileName().toString().endsWith(PRIMARY_EXTENSION);
    }

    @Override
    public boolean supportsUpload(String modelFileName, String labelsFileName) {
        if (modelFileName == null || labelsFileName == null) return false;
        // Check if model file is a zip and labels is txt
        // A stricter check could involve checking if modelFileName contains .mlpackage.zip
        return modelFileName.endsWith(UPLOAD_EXTENSION) && labelsFileName.endsWith("-labels.txt");
    }

    @Override
    public Optional<String> validateUpload(UploadedFile modelFile, UploadedFile labelsFile) {
        String modelExtension = modelFile.extension().toLowerCase();
        String labelsExtension = labelsFile.extension().toLowerCase();

        if (!modelExtension.equals(UPLOAD_EXTENSION)) {
            return Optional.of("Invalid model file type. Expected '" + UPLOAD_EXTENSION + "' but got '" + modelExtension + "' for ML Package");
        }
        if (!labelsExtension.equals(".txt")) {
            return Optional.of("Invalid labels file type. Expected '.txt' but got '" + labelsExtension + "'");
        }
        // Could add zip file validation here (e.g., check if it's a valid zip archive)
        return Optional.empty();
    }

    @Override
    public void verifyNames(String modelFileName, String labelsFileName) throws IllegalArgumentException {
        // Expects modelFileName to be like name-w-h-v.mlpackage.zip
        CoreMLPackageModel.verifyNames(modelFileName, labelsFileName);
    }

    @Override
    public Model loadFromPath(Path path, File modelsDirectory) throws IOException, IllegalArgumentException {
        // CoreMLPackageModel constructor handles loading from the directory path
        return new CoreMLPackageModel(path, modelsDirectory);
    }

    @Override
    public void saveUploadedFiles(UploadedFile modelFile, UploadedFile labelsFile, File modelsDirectory) throws IOException {
        Path labelsDestPath = modelsDirectory.toPath().resolve(labelsFile.filename());

        // Save labels file first
        logger.info("Saving labels CoreML package to: " + labelsDestPath);
        try (InputStream in = labelsFile.content(); OutputStream out = Files.newOutputStream(labelsDestPath)) {
            in.transferTo(out);
        } catch (IOException e) {
            throw new IOException("Failed to save CoreML Package labels file: " + labelsDestPath, e);
        }

        // Unzip the model file (which should be a zip archive)
        String targetDirName = modelFile.filename().replace(UPLOAD_EXTENSION, ""); // e.g., mymodel.mlpackage
        Path targetDirPath = modelsDirectory.toPath().resolve(targetDirName);

        logger.info("Unzipping CoreML package " + modelFile.filename() + " to: " + targetDirPath);

        // Ensure target directory exists (will create if not)
        Files.createDirectories(targetDirPath);

        try (InputStream fis = modelFile.content();
             ZipInputStream zis = new ZipInputStream(fis)) {

            ZipEntry zipEntry = zis.getNextEntry();
            while (zipEntry != null) {
                Path newPath = resolveAndValidateZipPath(targetDirPath, zipEntry.getName());
                if (zipEntry.isDirectory()) {
                    Files.createDirectories(newPath);
                } else {
                    // Create parent directories if necessary
                    if (newPath.getParent() != null) {
                        if (Files.notExists(newPath.getParent())) {
                            Files.createDirectories(newPath.getParent());
                        }
                    }
                    // Write file content
                    try (FileOutputStream fos = new FileOutputStream(newPath.toFile())) {
                        byte[] buffer = new byte[1024];
                        int len;
                        while ((len = zis.read(buffer)) > 0) {
                            fos.write(buffer, 0, len);
                        }
                    }
                }
                zis.closeEntry();
                zipEntry = zis.getNextEntry();
            }
             logger.info("Successfully unzipped to: " + targetDirPath);
        } catch (IOException e) {
            // Clean up labels file and potentially partially unzipped directory
            try { Files.deleteIfExists(labelsDestPath); } catch (IOException ignored) {}
            // Add more robust cleanup if needed (e.g., FileUtils.deleteDirectory)
            if (Files.exists(targetDirPath)) {
                logger.error("Failed to unzip CoreML package. Partially unzipped directory might remain: " + targetDirPath);
            }
            throw new IOException("Failed to unzip CoreML package: " + modelFile.filename(), e);
        }
    }

    // Helper method to prevent Zip Slip vulnerability
    private Path resolveAndValidateZipPath(Path targetDir, String entryName) throws IOException {
        // Normalize the entry name and resolve it against the target directory
        Path entryPath = targetDir.resolve(entryName).normalize();

        // Check if the resolved path is still within the target directory
        if (!entryPath.startsWith(targetDir)) {
            throw new IOException("Zip entry is outside of the target dir: " + entryName);
        }
        return entryPath;
    }
} 