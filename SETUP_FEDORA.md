# Fedora Setup for AI Engineering

เอกสารนี้เตรียมเครื่องสำหรับ Batch 01 โดยหลีกเลี่ยงการล็อก CUDA/PyTorch version แบบตายตัวก่อนถึงบทที่ต้องใช้ GPU

## 1. Update system

```bash
sudo dnf upgrade --refresh
```

## 2. Development packages

```bash
sudo dnf install -y \
  git gcc gcc-c++ make cmake \
  python3 python3-pip python3-devel
```

ตรวจ:

```bash
git --version
gcc --version
g++ --version
python3 --version
```

## 3. Clone course

```bash
git clone https://github.com/TxngJr/MASTER-PROMPT-ZERO-TO-ELITE-AI-ENGINEER.git
cd MASTER-PROMPT-ZERO-TO-ELITE-AI-ENGINEER
```

## 4. Python virtual environment

อย่าติดตั้ง package ของ course ลง system Python โดยตรง

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-batch01.txt
```

ตรวจ interpreter:

```bash
which python
python -c "import sys; print(sys.executable)"
```

## 5. Jupyter

```bash
jupyter lab
```

## 6. NVIDIA verification

ตรวจของเดิมก่อน:

```bash
lspci | grep -Ei 'vga|3d|display'
nvidia-smi
```

ถ้า `nvidia-smi` ใช้งานได้อยู่แล้ว อย่าเปลี่ยน driver โดยไม่มีเหตุผล

ถ้ายังไม่มี NVIDIA driver ให้ใช้ RPM Fusion ที่ตรงกับ Fedora รุ่นปัจจุบันและตรวจขั้นตอน Secure Boot/MOK ของรุ่นนั้นก่อนติดตั้ง kernel module

## 7. PyTorch policy

Batch 01 ยังไม่ต้องใช้ PyTorch เมื่อถึงบท PyTorch ให้เลือกคำสั่งล่าสุดจาก official selector:

https://pytorch.org/get-started/locally/

หลังติดตั้งในอนาคต:

```python
import torch

print(torch.__version__)
print(torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
```

## 8. Hardware-aware workflow

- เริ่ม dataset/model จากขนาดเล็ก
- วัด RAM/VRAM ก่อน scale
- รองรับ CPU fallback
- ไม่ commit dataset/output ใหญ่
- ปิด process ที่กิน VRAM โดยไม่จำเป็น

```bash
free -h
df -h
nvidia-smi
ps aux --sort=-%mem | head
```

## 9. Run Batch 01 tests

```bash
source .venv/bin/activate
pytest -q
```

ถ้า test ไม่ผ่าน เปิด [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
