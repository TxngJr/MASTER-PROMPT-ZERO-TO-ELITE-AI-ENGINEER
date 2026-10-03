# Chapter 08 Solutions

1. `sqrt(Σ(x-z)^2)`
2. `Σ|x-z|`
3. k เล็ก → variance สูง / bias ต่ำโดยทั่วไป
4. ไม่ให้ feature scale ใหญ่ dominate distance
5. ส่วนใหญ่เก็บ training instances/index structure

6. เพิ่ม noise dimensions ทำ geometry ของ distance แย่ลง
7. space โตเร็วและ points sparse เมื่อ dimension สูง
8. uniform นับ vote เท่ากัน; distance ให้จุดใกล้ weight มากกว่า
9. O(n_train × d) ต่อ query สำหรับ brute force
10. tree pruning ลดประโยชน์เมื่อ high-dimensional neighborhoods ไม่แยกชัด

ข้อ coding/challenge ควร implement พร้อม test และ benchmark เอง
