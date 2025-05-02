/*
 * Copyright (C) Photon Vision.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

package org.photonvision.model.manager;

import java.util.LinkedList;
import java.util.stream.Stream;
import org.junit.jupiter.params.provider.Arguments;

/** Tests for the CoreMLPackageManager implementation. */
public class CoreMLPackageManagerTest extends ModelManagerTest {
    private static final CoreMLPackageManager manager = new CoreMLPackageManager();

    private static LinkedList<String[]> passNames =
            new LinkedList<String[]>(
                    java.util.Arrays.asList(
                            new String[] {
                                "note-640-640-yolov5s.mlpackage.zip", "note-640-640-yolov5s-labels.txt"
                            },
                            new String[] {
                                "object-640-640-yolov8n.mlpackage.zip", "object-640-640-yolov8n-labels.txt"
                            },
                            new String[] {
                                "example_1.2-640-640-yolov5l.mlpackage.zip",
                                "example_1.2-640-640-yolov5l-labels.txt"
                            },
                            new String[] {
                                "demo_3.5-640-640-yolov8m.mlpackage.zip", "demo_3.5-640-640-yolov8m-labels.txt"
                            },
                            new String[] {
                                "sample-640-640-yolov5x.mlpackage.zip", "sample-640-640-yolov5x-labels.txt"
                            },
                            new String[] {
                                "test_case-640-640-yolov8s.mlpackage.zip", "test_case-640-640-yolov8s-labels.txt"
                            },
                            new String[] {
                                "model_ABC-640-640-yolov5n.mlpackage.zip", "model_ABC-640-640-yolov5n-labels.txt"
                            },
                            new String[] {
                                "my_model-640-640-yolov8x.mlpackage.zip", "my_model-640-640-yolov8x-labels.txt"
                            },
                            new String[] {
                                "name_1.0-640-640-yolov5n.mlpackage.zip", "name_1.0-640-640-yolov5n-labels.txt"
                            },
                            new String[] {
                                "valid_name-640-640-yolov8s.mlpackage.zip", "valid_name-640-640-yolov8s-labels.txt"
                            },
                            new String[] {
                                "test.model-640-640-yolov5l.mlpackage.zip", "test.model-640-640-yolov5l-labels.txt"
                            },
                            new String[] {
                                "case1_test-640-640-yolov8m.mlpackage.zip", "case1_test-640-640-yolov8m-labels.txt"
                            },
                            new String[] {
                                "A123-640-640-yolov5x.mlpackage.zip", "A123-640-640-yolov5x-labels.txt"
                            },
                            new String[] {
                                "z_y_test.model-640-640-yolov8n.mlpackage.zip",
                                "z_y_test.model-640-640-yolov8n-labels.txt"
                            }));

    private static LinkedList<String[]> parsedPassNames =
            new LinkedList<String[]>(
                    java.util.Arrays.asList(
                            new String[] {"note", "640", "640", "yolov5s"},
                            new String[] {"object", "640", "640", "yolov8n"},
                            new String[] {"example_1.2", "640", "640", "yolov5l"},
                            new String[] {"demo_3.5", "640", "640", "yolov8m"},
                            new String[] {"sample", "640", "640", "yolov5x"},
                            new String[] {"test_case", "640", "640", "yolov8s"},
                            new String[] {"model_ABC", "640", "640", "yolov5n"},
                            new String[] {"my_model", "640", "640", "yolov8x"},
                            new String[] {"name_1.0", "640", "640", "yolov5n"},
                            new String[] {"valid_name", "640", "640", "yolov8s"},
                            new String[] {"test.model", "640", "640", "yolov5l"},
                            new String[] {"case1_test", "640", "640", "yolov8m"},
                            new String[] {"A123", "640", "640", "yolov5x"},
                            new String[] {"z_y_test.model", "640", "640", "yolov8n"}));

    private static LinkedList<String[]> failNames =
            new LinkedList<String[]>(
                    java.util.Arrays.asList(
                            new String[] {"note-yolov5s.mlpackage.zip", "note-640-640-yolov5s-labels.txt"},
                            new String[] {"640-640-yolov8n.mlpackage.zip", "object-640-640-yolov8n-labels.txt"},
                            new String[] {"example_1.2.mlpackage.zip", "example_1.2-640-640-yolov5l-labels.txt"},
                            new String[] {
                                "demo_3.5-640-yolov8m.mlpackage.zip", "demo_3.5-640-640-yolov8m-labels.txt"
                            },
                            new String[] {"sample-640.mlpackage.zip", "sample-640-640-yolov5x-labels.txt"},
                            new String[] {"test_case.txt", "test_case-640-640-yolov8s-labels.txt"},
                            new String[] {"model_ABC.onnx", "model_ABC-640-640-yolov5n-labels.txt"},
                            new String[] {"my_model", "my_model-640-640-yolov8x-labels.txt"},
                            new String[] {"name_1.0-yolov5n.mlpackage.zip", "wrong-labels.txt"},
                            new String[] {"", "valid_name-640-640-yolov8s-labels.txt"},
                            new String[] {null, "test.model-640-640-yolov5l-labels.txt"},
                            new String[] {"case1_test-640-640-yolov8m.mlpackage.zip", null},
                            new String[] {"A123-640-640.mlpackage.zip", "different-labels.txt"},
                            new String[] {"z_y_test.model", ""}));

    // 用于存储转换后的包目录名称，用于测试 parseModelName 方法
    private static LinkedList<String> packageDirs = new LinkedList<>();

    static {
        // 初始化 packageDirs，将 .zip 文件名转换为 .mlpackage 目录名
        for (String[] name : passNames) {
            packageDirs.add(name[0].replace(".zip", ""));
        }
    }

    @Override
    protected ModelManager getModelManager() {
        return manager;
    }

    @Override
    protected LinkedList<String[]> getValidNamePairs() {
        return passNames;
    }

    @Override
    protected LinkedList<String[]> getInvalidNamePairs() {
        return failNames;
    }

    @Override
    protected LinkedList<String[]> getParsedValidNames() {
        return parsedPassNames;
    }

    /** Provides the valid name pairs for testing name validation. */
    static Stream<Arguments> verifyPassNameProvider() {
        return passNames.stream().map(array -> Arguments.of((Object) array));
    }

    /** Provides the invalid name pairs for testing name validation failures. */
    static Stream<Arguments> verifyFailNameProvider() {
        return failNames.stream().map(array -> Arguments.of((Object) array));
    }

    /**
     * Provides the test cases for name parsing verification. 对于 CoreMLPackageManager，我们需要将 zip
     * 文件名转换为目录名格式， 因为 parseModelName 方法期望输入的是 .mlpackage 目录名
     */
    static Stream<Arguments> parseNameProvider() {
        // 返回解析后的名称和包目录名（非zip文件名）
        return java.util.stream.IntStream.range(0, passNames.size())
                .mapToObj(i -> Arguments.of(parsedPassNames.get(i), packageDirs.get(i)));
    }
}
