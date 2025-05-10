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

package org.photonvision.common.hardware.metrics.cmds;

import org.photonvision.common.configuration.HardwareConfig;

public class MacOSCmds extends CmdBase {
    public void initCmds(HardwareConfig config) {
        cpuMemoryCommand = "sysctl -n hw.memsize | awk '{print $1/1024/1024}'";

        cpuTemperatureCommand = "ioreg -rc 'AppleSmartBattery' | grep -i \"\\\"Temperature\\\"\" | head -n 1 | awk '{print ($3 / 100)}' | tr -d '\\n' || echo 'N/A'";
        
        cpuUtilizationCommand = 
                "top -l 1 | grep -E '^CPU' | awk '{print 100-$7}'";

        cpuUptimeCommand = "system_profiler SPSoftwareDataType -detailLevel mini | grep 'Time since boot' | awk -F': ' '{print $2}'";
        
        ramUsageCommand = "ps -A -o rss | awk '{sum+=$1} END {print int(sum/1024)}'";
        
        diskUsageCommand = "df -h . | awk 'NR==2 {print $5}'";
    }
} 