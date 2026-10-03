# Troubleshooting

ใช้หลัก **observe before change**

## Python
```bash
python3 --version
which python3
which python
python -m pip --version
python -m pip list
```

ถ้า `python` กับ `pip` อยู่คนละ environment ให้ activate `.venv` ใหม่

## CSV/path
```bash
pwd
find . -maxdepth 3 -type f | sort
```
ใน Python ใช้ `pathlib.Path`

## Git
```bash
git status
git diff
git log --oneline --decorate -10
git remote -v
```

อย่าใช้ `git reset --hard` แบบเดาสุ่ม

## NVIDIA
```bash
nvidia-smi
lsmod | grep nvidia
uname -r
```

## Debug loop

1. อ่าน error บรรทัดสุดท้าย
2. หา frame แรกที่เป็น code ของเรา
3. ตรวจ type/shape/value
4. ลด input ให้เล็กที่สุดที่ยังทำให้พัง
5. แก้ root cause
6. เพิ่ม regression test
