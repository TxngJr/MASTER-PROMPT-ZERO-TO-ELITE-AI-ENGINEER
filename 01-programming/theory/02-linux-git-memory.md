# Linux, Bash, Git and Memory Foundations

## Program vs Process

**Program** คือ code/executable บน storage

**Process** คือ instance ที่กำลังทำงาน พร้อม state เช่น:

- process id (PID)
- virtual address space
- open files
- environment
- threads

ดู process:

```bash
ps aux | head
ps -ef | grep python
```

## Exit status

Unix command คืน exit code

```bash
python script.py
echo $?
```

โดย convention `0` มักหมายถึง success

## stdin / stdout / stderr

```bash
python app.py > output.txt
python app.py 2> error.txt
```

pipe:

```bash
find . -type f | sort | head
```

output ของ command หนึ่งเป็น input ให้อีก command

## Environment Variables

```bash
export APP_MODE=development
echo "$APP_MODE"
```

Python:

```python
import os
print(os.environ.get("APP_MODE", "unknown"))
```

อย่า commit secret/API key ลง Git

## PATH

`PATH` เป็นรายการ directories ที่ shell ใช้หา executable

```bash
echo "$PATH"
which python
which git
```

นี่อธิบาย bug ยอดนิยม: install package ด้วย pip ของ Python A แต่รัน Python B

## Permissions

```bash
ls -l
```

แนวคิด owner/group/other และ read/write/execute

อย่าใช้ `chmod 777` เป็น default fix

## Git as a graph

แต่ละ commit อ้าง parent commit และ snapshot/tree ของ project

```text
A ← B ← C  main
     \
      D ← E feature
```

commands:

```bash
git status
git log --oneline --graph --decorate --all
git diff
git diff --staged
```

## Branch workflow

```bash
git switch -c feature/example
# edit
git add path/to/file
git diff --staged
git commit -m "feat: explain example"
```

commit message บอก intent ไม่ใช่แค่ "update"

## C memory mental model

C ให้ programmer จัดการ memory ใกล้ hardware มากกว่า Python

```c
#include <stdio.h>

int main(void) {
    int x = 42;
    int *p = &x;

    printf("%d\n", *p);
    return 0;
}
```

- `&x` = address ของ x
- `p` = pointer ที่เก็บ address
- `*p` = dereference อ่านค่าที่ address นั้น

## Stack and Heap

ตัวอย่าง C:

```c
#include <stdlib.h>

int main(void) {
    int local = 1;
    int *data = malloc(100 * sizeof(int));

    if (data == NULL) {
        return 1;
    }

    data[0] = local;
    free(data);
    return 0;
}
```

mental model:
- `local` เป็น automatic-storage local object
- `malloc` ขอ dynamic storage
- `free` คืน storage

ถ้าลืม `free` อาจเกิด memory leak

## C++ RAII

C++ นิยมผูก resource lifetime กับ object lifetime:

```cpp
#include <vector>

int main() {
    std::vector<float> values(100);
    values[0] = 1.0f;
}
```

`std::vector` จัดการ dynamic memory ให้โดยใช้ RAII จึงปลอดภัยกว่า raw `malloc/free` ในงานทั่วไป

## Why AI engineers need this

ต่อไปคุณจะเจอ:

- CPU RAM
- GPU VRAM
- tensors
- pinned memory
- buffers
- memory mapped datasets
- process workers

ถ้าไม่มี mental model เรื่อง resource และ lifetime จะ debug out-of-memory ยากมาก
