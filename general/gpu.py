import torch
import comfy.model_management
from ..core import logger
import os
import platform

def is_jetson() -> bool:
    """
    Determines if the Python environment is running on a Jetson device by checking the device model
    information or the platform release.
    """
    PROC_DEVICE_MODEL = ''
    try:
        with open('/proc/device-tree/model', 'r') as f:
            PROC_DEVICE_MODEL = f.read().strip()
            logger.info(f"Device model: {PROC_DEVICE_MODEL}")
            return "NVIDIA" in PROC_DEVICE_MODEL
    except Exception as e:
        # logger.warning(f"JETSON: Could not read /proc/device-tree/model: {e} (If you're not using Jetson, ignore this warning)")
        # If /proc/device-tree/model is not available, check platform.release()
        platform_release = platform.release()
        logger.info(f"Platform release: {platform_release}")
        if 'tegra' in platform_release.lower():
            logger.info("Detected 'tegra' in platform release. Assuming Jetson device.")
            return True
        else:
            logger.info("JETSON: Not detected.")
            return False

def is_rocm() -> bool:
    """
    判断当前 PyTorch 是否为 AMD ROCm(HIP) 构建。
    ROCm 版 PyTorch 会把 HIP 映射到 CUDA API，torch.version.hip 不为 None 即代表 AMD 环境。
    """
    try:
        return getattr(torch.version, 'hip', None) is not None
    except Exception:
        return False

IS_JETSON = is_jetson()

class CGPUInfo:
    """
    This class is responsible for getting information from GPU (ONLY).
    支持三类后端：NVIDIA(pynvml)、Jetson(jtop)、AMD ROCm(amdsmi / pyamdgpuinfo / torch 兜底)。
    """
    cuda = False
    pynvmlLoaded = False
    jtopLoaded = False
    # AMD 相关标志
    amdsmiLoaded = False
    pyamdgpuinfoLoaded = False
    amdBackend = None  # 取值：'amdsmi' | 'pyamdgpuinfo' | 'torch'
    amdsmi = None
    amdsmiHandles = None
    pyamdgpuinfo = None
    cudaAvailable = False
    torchDevice = 'cpu'
    cudaDevice = 'cpu'
    cudaDevicesFound = 0
    switchGPU = True
    switchVRAM = True
    switchTemperature = True
    gpus = []
    gpusUtilization = []
    gpusVRAM = []
    gpusTemperature = []

    def __init__(self):
        if IS_JETSON:
            # Try to import jtop for Jetson devices
            try:
                from jtop import jtop
                self.jtopInstance = jtop()
                self.jtopInstance.start()
                self.jtopLoaded = True
                logger.info('jtop initialized on Jetson device.')
            except ImportError as e:
                logger.error('jtop is not installed. ' + str(e))
            except Exception as e:
                logger.error('Could not initialize jtop. ' + str(e))
        else:
            # Try to import pynvml for non-Jetson devices
            try:
                import pynvml
                self.pynvml = pynvml
                self.pynvml.nvmlInit()
                self.pynvmlLoaded = True
                logger.info('pynvml (NVIDIA) initialized.')
            except ImportError as e:
                logger.error('pynvml is not installed. ' + str(e))
            except Exception as e:
                logger.error('Could not init pynvml (NVIDIA). ' + str(e))

        # NVIDIA 与 Jetson 都不可用、且为 AMD ROCm 环境时，尝试 AMD 监控后端
        if not self.pynvmlLoaded and not self.jtopLoaded and is_rocm():
            self._init_amd()

        self.anygpuLoaded = (
            self.pynvmlLoaded
            or self.jtopLoaded
            or self.amdsmiLoaded
            or self.pyamdgpuinfoLoaded
            or self.amdBackend == 'torch'
        )

        try:
            self.torchDevice = comfy.model_management.get_torch_device_name(comfy.model_management.get_torch_device())
        except Exception as e:
            logger.error('Could not pick default device. ' + str(e))

        # 后端无关的 0 卡检测：任何后端报告 0 张卡都禁用监控
        if self.anygpuLoaded and self.deviceGetCount() == 0:
            logger.warning('No GPU detected, disabling GPU monitoring.')
            self.anygpuLoaded = False
            self.pynvmlLoaded = False
            self.jtopLoaded = False
            self.amdsmiLoaded = False
            self.pyamdgpuinfoLoaded = False
            self.amdBackend = None

        if self.anygpuLoaded:
            if self.deviceGetCount() > 0:
                self.cudaDevicesFound = self.deviceGetCount()

                logger.info(f"GPU/s:")

                for deviceIndex in range(self.cudaDevicesFound):
                    deviceHandle = self.deviceGetHandleByIndex(deviceIndex)

                    gpuName = self.deviceGetName(deviceHandle, deviceIndex)

                    logger.info(f"{deviceIndex}) {gpuName}")

                    self.gpus.append({
                        'index': deviceIndex,
                        'name': gpuName,
                    })

                    # Same index as gpus, with default values
                    self.gpusUtilization.append(True)
                    self.gpusVRAM.append(True)
                    self.gpusTemperature.append(True)

                self.cuda = True
                logger.info(self.systemGetDriverVersion())
            else:
                logger.warning('No GPU with CUDA detected.')
        else:
            logger.warning('No GPU monitoring libraries available.')

        self.cudaDevice = 'cpu' if self.torchDevice == 'cpu' else 'cuda'
        self.cudaAvailable = torch.cuda.is_available()

        if self.cuda and self.cudaAvailable and self.torchDevice == 'cpu':
            logger.warning('CUDA is available, but torch is using CPU.')

    def _init_amd(self):
        """
        初始化 AMD GPU 监控后端，优先级：
        amdsmi(AMD 官方，指标最全) -> pyamdgpuinfo(Linux sysfs) -> torch(仅显存兜底)。
        任一后端初始化成功即返回；所有 import/调用失败都被捕获，优雅降级。
        """
        # 后端 A：AMD 官方 amd-smi 库（Linux/Windows，含利用率、温度、显存、卡名）
        try:
            import amdsmi
            amdsmi.amdsmi_init()
            # 不同版本 amdsmi_get_processor_handles 签名可能不同，做兼容处理
            try:
                handles = amdsmi.amdsmi_get_processor_handles()
            except TypeError:
                proc_enum = getattr(amdsmi, 'AmdSmiProcessor', None) or getattr(amdsmi, 'AmdSmiProcessorType', None)
                handles = amdsmi.amdsmi_get_processor_handles(proc_enum.GPU) if proc_enum is not None else []
            if handles:
                self.amdsmi = amdsmi
                self.amdsmiHandles = handles
                self.amdsmiLoaded = True
                self.amdBackend = 'amdsmi'
                logger.info('amdsmi (AMD ROCm) initialized.')
                return
            logger.warning('amdsmi initialized but no AMD GPU handle found.')
        except ImportError as e:
            logger.error('amdsmi is not installed. ' + str(e))
        except Exception as e:
            logger.error('Could not init amdsmi (AMD). ' + str(e))

        # 后端 B：pyamdgpuinfo（仅 Linux，基于 sysfs/libdrm，含利用率、温度、显存、卡名）
        try:
            import pyamdgpuinfo
            if pyamdgpuinfo.detect_gpus() > 0:
                self.pyamdgpuinfo = pyamdgpuinfo
                self.pyamdgpuinfoLoaded = True
                self.amdBackend = 'pyamdgpuinfo'
                logger.info('pyamdgpuinfo (AMD) initialized.')
                return
            logger.warning('pyamdgpuinfo detected no AMD GPU.')
        except ImportError as e:
            logger.error('pyamdgpuinfo is not installed. ' + str(e))
        except Exception as e:
            logger.error('Could not init pyamdgpuinfo (AMD). ' + str(e))

        # 后端 C：无监控库时，若 torch 能看到 ROCm 设备，则用 torch 兜底（至少能显示显存）
        try:
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                self.amdBackend = 'torch'
                logger.info('No AMD monitoring library found, falling back to torch (VRAM only).')
        except Exception as e:
            logger.error('AMD torch fallback check failed. ' + str(e))

    def _amdTorchName(self, deviceIndex):
        """ROCm 下通过 torch 获取显卡名（作为 AMD 各后端的兜底）"""
        try:
            name = torch.cuda.get_device_name(deviceIndex)
            if name:
                return name
        except Exception as e:
            logger.error('Could not get AMD GPU name from torch. ' + str(e))
        return 'AMD GPU'

    def _amdTorchMemoryInfo(self, deviceIndex):
        """torch 兜底显存信息（ROCm 下 torch.cuda 可用；used 仅统计 torch 进程占用）"""
        try:
            total = torch.cuda.get_device_properties(deviceIndex).total_memory
            used = torch.cuda.memory_allocated(deviceIndex)
            return {'total': total, 'used': used}
        except Exception as e:
            logger.error('Could not get AMD VRAM info from torch. ' + str(e))
            return {'total': 1, 'used': 1}

    def getInfo(self):
        logger.debug('Getting GPUs info...')
        return self.gpus

    def getStatus(self):
        gpuUtilization = -1
        gpuTemperature = -1
        vramUsed = -1
        vramTotal = -1
        vramPercent = -1

        gpuType = ''
        gpus = []

        if self.cudaDevice == 'cpu':
            gpuType = 'cpu'
            gpus.append({
                'gpu_utilization': -1,
                'gpu_temperature': -1,
                'vram_total': -1,
                'vram_used': -1,
                'vram_used_percent': -1,
            })
        else:
            gpuType = self.cudaDevice

            if self.anygpuLoaded and self.cuda and self.cudaAvailable:
                for deviceIndex in range(self.cudaDevicesFound):
                    deviceHandle = self.deviceGetHandleByIndex(deviceIndex)

                    gpuUtilization = -1
                    vramPercent = -1
                    vramUsed = -1
                    vramTotal = -1
                    gpuTemperature = -1

                    # GPU Utilization
                    if self.switchGPU and self.gpusUtilization[deviceIndex]:
                        try:
                            gpuUtilization = self.deviceGetUtilizationRates(deviceHandle)
                        except Exception as e:
                            logger.error('Could not get GPU utilization. ' + str(e))
                            logger.error('Monitor of GPU is turning off.')
                            self.switchGPU = False

                    if self.switchVRAM and self.gpusVRAM[deviceIndex]:
                        try:
                            memory = self.deviceGetMemoryInfo(deviceHandle, deviceIndex)
                            vramUsed = memory['used']
                            vramTotal = memory['total']

                            # Check if vramTotal is not zero or None
                            if vramTotal and vramTotal != 0:
                                vramPercent = vramUsed / vramTotal * 100
                        except Exception as e:
                            logger.error('Could not get GPU memory info. ' + str(e))
                            self.switchVRAM = False

                    # Temperature
                    if self.switchTemperature and self.gpusTemperature[deviceIndex]:
                        try:
                            gpuTemperature = self.deviceGetTemperature(deviceHandle)
                        except Exception as e:
                            logger.error('Could not get GPU temperature. Turning off this feature. ' + str(e))
                            self.switchTemperature = False

                    gpus.append({
                        'gpu_utilization': gpuUtilization,
                        'gpu_temperature': gpuTemperature,
                        'vram_total': vramTotal,
                        'vram_used': vramUsed,
                        'vram_used_percent': vramPercent,
                    })

        return {
            'device_type': gpuType,
            'gpus': gpus,
        }

    def deviceGetCount(self):
        if self.pynvmlLoaded:
            return self.pynvml.nvmlDeviceGetCount()
        elif self.jtopLoaded:
            # For Jetson devices, we assume there's one GPU
            return 1
        elif self.amdsmiLoaded:
            return len(self.amdsmiHandles)
        elif self.pyamdgpuinfoLoaded:
            return self.pyamdgpuinfo.detect_gpus()
        elif self.amdBackend == 'torch':
            return torch.cuda.device_count()
        else:
            return 0

    def deviceGetHandleByIndex(self, index):
        if self.pynvmlLoaded:
            return self.pynvml.nvmlDeviceGetHandleByIndex(index)
        elif self.jtopLoaded:
            return index  # On Jetson, index acts as handle
        elif self.amdsmiLoaded:
            return self.amdsmiHandles[index]
        elif self.pyamdgpuinfoLoaded:
            return self.pyamdgpuinfo.get_gpu(index)
        elif self.amdBackend == 'torch':
            return index
        else:
            return 0

    def deviceGetName(self, deviceHandle, deviceIndex):
        if self.pynvmlLoaded:
            gpuName = 'Unknown GPU'

            try:
                gpuName = self.pynvml.nvmlDeviceGetName(deviceHandle)
                try:
                    gpuName = gpuName.decode('utf-8', errors='ignore')
                except AttributeError:
                    pass

            except UnicodeDecodeError as e:
                gpuName = 'Unknown GPU (decoding error)'
                logger.error(f"UnicodeDecodeError: {e}")

            return gpuName
        elif self.jtopLoaded:
            # Access the GPU name from self.jtopInstance.gpu
            try:
                gpu_info = self.jtopInstance.gpu
                gpu_name = next(iter(gpu_info.keys()))
                return gpu_name
            except Exception as e:
                logger.error('Could not get GPU name. ' + str(e))
                return 'Unknown GPU'
        elif self.amdsmiLoaded:
            try:
                board = self.amdsmi.amdsmi_get_gpu_board_info(deviceHandle)
                name = board.get('product_name')
                if name and name != 'N/A':
                    return name
            except Exception as e:
                logger.error('Could not get AMD GPU name from amdsmi. ' + str(e))
            return self._amdTorchName(deviceIndex)
        elif self.pyamdgpuinfoLoaded:
            try:
                name = getattr(deviceHandle, 'name', None)
                if name:
                    return name
            except Exception as e:
                logger.error('Could not get AMD GPU name from pyamdgpuinfo. ' + str(e))
            return self._amdTorchName(deviceIndex)
        elif self.amdBackend == 'torch':
            return self._amdTorchName(deviceIndex)
        else:
            return ''

    def systemGetDriverVersion(self):
        if self.pynvmlLoaded:
            return f'NVIDIA Driver: {self.pynvml.nvmlSystemGetDriverVersion()}'
        elif self.jtopLoaded:
            # No direct method to get driver version from jtop
            return 'NVIDIA Driver: unknown'
        elif self.amdsmiLoaded or self.pyamdgpuinfoLoaded or self.amdBackend == 'torch':
            hip = getattr(torch.version, 'hip', None)
            return f'AMD ROCm/HIP: {hip}' if hip else 'AMD Driver: unknown'
        else:
            return 'Driver unknown'

    def deviceGetUtilizationRates(self, deviceHandle):
        if self.pynvmlLoaded:
            return self.pynvml.nvmlDeviceGetUtilizationRates(deviceHandle).gpu
        elif self.jtopLoaded:
            # GPU utilization from jtop stats
            try:
                gpu_util = self.jtopInstance.stats.get('GPU', -1)
                return gpu_util
            except Exception as e:
                logger.error('Could not get GPU utilization. ' + str(e))
                return -1
        elif self.amdsmiLoaded:
            try:
                metrics = self.amdsmi.amdsmi_get_gpu_metrics_info(deviceHandle)
                util = metrics.get('average_gfx_activity')
                # 无效值（None 或越界）返回 -1（不可用哨兵）
                if util is None or util > 100:
                    return -1
                return util
            except Exception as e:
                logger.error('Could not get AMD GPU utilization. ' + str(e))
                return -1
        elif self.pyamdgpuinfoLoaded:
            try:
                load = deviceHandle.query_load()
                if load is None or load > 100:
                    return -1
                return load
            except Exception as e:
                logger.error('Could not get AMD GPU utilization. ' + str(e))
                return -1
        elif self.amdBackend == 'torch':
            # torch 无法提供整卡利用率
            return -1
        else:
            return 0

    def deviceGetMemoryInfo(self, deviceHandle, deviceIndex=None):
        if self.pynvmlLoaded:
            mem = self.pynvml.nvmlDeviceGetMemoryInfo(deviceHandle)
            return {'total': mem.total, 'used': mem.used}
        elif self.jtopLoaded:
            mem_data = self.jtopInstance.memory['RAM']
            total = mem_data['tot']
            used = mem_data['used']
            return {'total': total, 'used': used}
        elif self.amdsmiLoaded:
            try:
                vram = self.amdsmi.amdsmi_get_gpu_vram_usage(deviceHandle)
                # amdsmi 返回单位为 MB，转换为字节以与其它后端保持一致
                total = (vram.get('vram_total') or 0) * 1024 * 1024
                used = (vram.get('vram_used') or 0) * 1024 * 1024
                if total:
                    return {'total': total, 'used': used}
            except Exception as e:
                logger.error('Could not get AMD VRAM info from amdsmi. ' + str(e))
            return self._amdTorchMemoryInfo(deviceIndex)
        elif self.pyamdgpuinfoLoaded:
            try:
                total = deviceHandle.memory_info.get('vram_size') if deviceHandle.memory_info else None
                used = deviceHandle.query_vram_usage()
                if total:
                    return {'total': total, 'used': used}
            except Exception as e:
                logger.error('Could not get AMD VRAM info from pyamdgpuinfo. ' + str(e))
            return self._amdTorchMemoryInfo(deviceIndex)
        elif self.amdBackend == 'torch':
            return self._amdTorchMemoryInfo(deviceIndex)
        else:
            return {'total': 1, 'used': 1}

    def deviceGetTemperature(self, deviceHandle):
        if self.pynvmlLoaded:
            return self.pynvml.nvmlDeviceGetTemperature(deviceHandle, self.pynvml.NVML_TEMPERATURE_GPU)
        elif self.jtopLoaded:
            try:
                temperature = self.jtopInstance.stats.get('Temp gpu', -1)
                return temperature
            except Exception as e:
                logger.error('Could not get GPU temperature. ' + str(e))
                return -1
        elif self.amdsmiLoaded:
            try:
                metrics = self.amdsmi.amdsmi_get_gpu_metrics_info(deviceHandle)
                temp = metrics.get('temperature_edge')
                # 边缘温度无效时回退到热点温度
                if temp is None or temp > 200:
                    temp = metrics.get('temperature_hotspot')
                if temp is None or temp > 200:
                    return -1
                return temp
            except Exception as e:
                logger.error('Could not get AMD GPU temperature. ' + str(e))
                return -1
        elif self.pyamdgpuinfoLoaded:
            try:
                temp = deviceHandle.query_temperature()  # pyamdgpuinfo 已返回摄氏度
                if temp is None:
                    return -1
                return int(round(temp))
            except Exception as e:
                logger.error('Could not get AMD GPU temperature. ' + str(e))
                return -1
        elif self.amdBackend == 'torch':
            # torch 无法提供 GPU 温度
            return -1
        else:
            return 0

    def close(self):
        if self.jtopLoaded and self.jtopInstance is not None:
            self.jtopInstance.close()
        if self.amdsmiLoaded and self.amdsmi is not None:
            try:
                self.amdsmi.amdsmi_shut_down()
            except Exception as e:
                logger.error('Could not shut down amdsmi. ' + str(e))
