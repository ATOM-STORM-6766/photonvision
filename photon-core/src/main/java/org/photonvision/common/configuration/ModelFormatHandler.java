package org.photonvision.common.configuration;

import io.javalin.http.UploadedFile;
import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Optional;
import org.photonvision.vision.objects.Model;

/**
 * Interface defining the contract for handling a specific neural network model format.
 * Implementations of this interface encapsulate the logic for discovering, validating,
 * loading, and saving models of a particular type (e.g., RKNN files, CoreML packages).
 */
public interface ModelFormatHandler {

    /**
     * Data Transfer Object containing basic information about the model format for the frontend.
     */
    public static record Info(String name, String uploadAcceptType) {}

    /**
     * Gets the unique identifier name for this backend format.
     * This name is used internally and potentially in configuration/logging.
     * Example: "RKNN", "COREML_FILE", "COREML_PACKAGE"
     *
     * @return The backend name string.
     */
    String getBackendName();

    /**
     * Gets the string used for the HTML file input 'accept' attribute.
     * Example: ".rknn", ".mlmodel", ".zip"
     *
     * @return The accept type string.
     */
    String getUploadAcceptType();

    /**
     * Gets the associated {@link Model} implementation class for this format.
     *
     * @return The Class object of the Model implementation.
     */
    Class<? extends Model> getModelClass();

    /**
     * Checks if the given filesystem path (file or directory) is potentially supported by this handler.
     * Used during model discovery.
     *
     * @param path The path to check.
     * @return true if the path might be handled by this handler, false otherwise.
     */
    boolean supportsPath(Path path);

    /**
     * Checks if the given uploaded filenames potentially correspond to this model format.
     * This check should be based solely on filenames and expected extensions.
     *
     * @param modelFileName The filename of the uploaded model file (or zip for packages).
     * @param labelsFileName The filename of the uploaded labels file.
     * @return true if the filenames suggest this handler might be applicable, false otherwise.
     */
    boolean supportsUpload(String modelFileName, String labelsFileName);

    /**
     * Performs validation on the uploaded files themselves (beyond just filename checks).
     * This typically involves checking file extensions.
     *
     * @param modelFile The uploaded model file.
     * @param labelsFile The uploaded labels file.
     * @return An Optional containing an error message string if validation fails, or Optional.empty() if it passes.
     */
    Optional<String> validateUpload(UploadedFile modelFile, UploadedFile labelsFile);

    /**
     * Verifies that the model and labels filenames strictly adhere to the naming conventions
     * for this format and that their components match.
     *
     * @param modelFileName The model filename (or zip filename for packages).
     * @param labelsFileName The labels filename.
     * @throws IllegalArgumentException if the names are invalid or do not match.
     */
    void verifyNames(String modelFileName, String labelsFileName) throws IllegalArgumentException;

    /**
     * Loads a {@link Model} instance from the given filesystem path.
     *
     * @param path The path to the model file or directory.
     * @param modelsDirectory The root directory where models are stored (needed to find related files like labels).
     * @return The loaded Model instance.
     * @throws IOException If an error occurs during file reading.
     * @throws IllegalArgumentException If the model format is invalid or required files are missing.
     */
    Model loadFromPath(Path path, File modelsDirectory) throws IOException, IllegalArgumentException;

    /**
     * Saves the uploaded model and labels files to the specified directory.
     * For package formats, this includes unzipping the model archive.
     *
     * @param modelFile The uploaded model file (or zip archive).
     * @param labelsFile The uploaded labels file.
     * @param modelsDirectory The target directory to save the files.
     * @throws IOException If an error occurs during file writing or unzipping.
     */
    void saveUploadedFiles(UploadedFile modelFile, UploadedFile labelsFile, File modelsDirectory) throws IOException;

    /**
     * Gets the basic information DTO for this handler.
     *
     * @return The Info record.
     */
    Info getInfo();
} 