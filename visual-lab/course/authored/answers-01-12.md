# เฉลยที่ทบทวนเฉพาะข้อ — Foundations และ Classical ML

## 01
1. ชื่อ variable เป็น reference ที่ผูกกับ object ไม่ใช่สำเนาของ object: `a=[1]; b=a; b.append(2)` ทำให้ทั้ง a และ b เห็น `[1,2]`. แต่ `b=[9]` เปลี่ยน binding ของ b เท่านั้น a ยังเป็น `[1,2]`; ใช้กรณีนี้แยก mutation กับ reassignment.
2. Mutable เปลี่ยนเนื้อหา object เดิมได้ เช่น list.append ส่วน immutable เช่น int/tuple เปลี่ยนค่าโดยสร้าง/อ้าง object ใหม่. `a=(1,); b=a; b+=(2,)` ให้ a=(1,),b=(1,2); tuple ที่เก็บ list ยังมี list ภายในที่ mutable ได้.
3. `==` ตรวจความเท่ากันของค่า ส่วน `is` ตรวจ identity. `a=[1]; b=[1]` ให้ a==b เป็น True แต่ a is b เป็น False. ใช้ `x is None` ตรวจ sentinel; อย่าอาศัย interning ของเลข/strings เป็นตรรกะของโปรแกรม.
4. Virtual environment แยก interpreter/dependencies ของโครงงาน เพื่อไม่ให้การอัปเกรด library งานหนึ่งเปลี่ยนอีกงาน. ตรวจ `sys.executable` และ `python -m pip --version` ว่าอยู่ environment เดียวกัน แล้วบันทึก requirements; venv ไม่แยก kernel/OS เหมือน VM.
5. Commit อ้าง snapshot ของ tree, parent commit(s), author/committer และข้อความ พร้อม content-addressed objects. การแก้ไฟล์ใน working tree ยังไม่อยู่ใน commit จน stage/commit; ตรวจ `git diff` และ `git diff --staged` แยกสองสถานะก่อนบันทึก.
6. Default expression ถูกประเมินครั้งเดียวตอนสร้าง function จึงใช้ list เดิมข้าม calls. แก้เป็น `scores=None` แล้วสร้าง `scores=[]` ภายในเมื่อเป็น None. ทดสอบเรียก append_score(1) และ append_score(2) แยกกันต้องได้ [1] กับ [2]; ถ้าส่ง list เองจึงค่อยใช้ list นั้นตามสัญญา.
7. Generator เก็บสถานะการ iteration และผลิตสมาชิกทีละตัว จึงไม่จำเป็นต้องสร้างรายการ N ตัวพร้อมกัน. ผลรวมจาก generator ยังต้องอ่าน N ตัวและใช้ O(N) time แต่ accumulator ใช้ O(1) extra memory; ถ้าเรียก list(generator) จะกลับไปใช้ O(N) memory.
8. การกลืน exception ทำให้ผู้เรียกเห็นเหมือนสำเร็จทั้งที่ state ไม่ครบ. จับ exception ที่รู้วิธีรับมือ เช่น FileNotFoundError พร้อม path แล้วคืน error/raise ต่อ; ทดสอบไฟล์หายต้องไม่รายงาน row_count=0 ว่าเป็นไฟล์ปกติ.
9. PATH เป็นลำดับ directories ที่ shell ใช้ค้น executable เมื่อพิมพ์ชื่อที่ไม่มี slash. ถ้ามี python หลายตัว ตัวแรกที่เจอถูกใช้; ตรวจ `command -v python` และ `python -c 'import sys;print(sys.executable)'`. การเปลี่ยน current directory ไม่เท่ากับเปลี่ยน PATH.
10. Program คือชุดคำสั่ง/ไฟล์ ส่วน process คือ instance ที่ OS กำลังรัน มี PID, environment, current directory และ address space. รัน script เดียวสองครั้งเป็นสอง processes; error จาก relative path อาจเกิดเพราะ current directories ต่างกัน.
11. แปลง values เป็น list หนึ่งครั้ง ตรวจว่าไม่ว่าง แล้ว `return sum(values)/len(values)`; ถ้าว่าง raise ValueError. Expected: [2,4]→3.0, [0]→0.0, []→ValueError; กำหนดชัดว่าจะรับ generator และ reject nonfinite/nonnumeric อย่างไร.
12. ใช้ `for value in range(0,N+1,2): yield value` เมื่อกำหนด inclusive 0..N. Expected N=5→[0,2,4],N=0→[0],N=-1→[] หรือ reject ตาม contract ที่ประกาศ; ผลเป็น iterator ไม่ใช่ list จนกว่าจะ materialize.
13. ใช้ `sorted(Path(root).rglob('*.csv'))` แล้วกรอง is_file เพื่อไม่รวม directory ชื่อ .csv. สร้าง temporary directory ที่มี a.csv, nested/b.csv,c.txt แล้วตรวจว่าได้สองไฟล์ตามลำดับ deterministic; missing root ต้องมี policy ชัด ไม่รายงานว่า dataset ว่างเงียบ ๆ.
14. เรียงค่าตัวเลขเป็น a; n คี่ใช้ a[n//2], n คู่เฉลี่ย a[n//2-1:n//2+1]. [9,1,3]→3,[1,9]→5; [] ต้อง reject หรือรายงาน missing อย่างชัดเจน. บันทึกจำนวนค่าที่ parse ไม่ได้แยกจากจำนวนค่าที่ใช้คำนวณ.
15. เพิ่ม argparse argument `--limit` แบบ int ตรวจว่า ≥0 แล้วหยุดเมื่ออ่าน data rows ครบ N หลัง header. ไฟล์สามแถวกับ limit=2 ต้องรายงานสองแถว; limit=0 ไม่อ่าน data, ไม่มี flag อ่านครบ; อย่า off-by-one โดยนับ header เป็นแถวแรก.
16. Bug คือ scores=[] ถูกใช้ซ้ำข้าม calls. เปลี่ยน signature เป็น `def append_score(score,scores=None)` และสร้าง list ใหม่เมื่อ scores is None. Regression test ต้องเรียกสองครั้งโดยไม่ส่ง scores และตรวจว่า object ไม่ใช่ตัวเดียวกัน พร้อมกรณี caller ส่ง list ที่ตั้งใจให้ mutate.
17. `python -m pip` ใช้ pip module ของ interpreter ตัวที่เรียก จึงลดความเสี่ยงติดตั้งไปอีก Python เมื่อ executable pip ใน PATH ต่างกัน. ยืนยันจาก sys.executable และ pip --version; ไม่ควรแก้ด้วย sudo pip เพื่อกลบ environment mismatch.
18. ตรวจ current directory, resolved absolute path, environment/PATH, interpreter/venv, file permissions/user และตัวแปร config ที่กำหนด data root. บันทึกค่าจากทั้ง terminal A/B แล้วเปลี่ยนทีละปัจจัย; ข้อความ 'ไฟล์หาไม่เจอ' อย่างเดียวไม่พอระบุสาเหตุ.
19. ใช้ csv.DictReader อ่านทีละ row เพิ่ม row_count และ missing_count ต่อ column ด้วยกฎ missing ที่ระบุ. สำหรับ `a,b\n1,\n,2\n` ได้สองแถวและ missing(a)=missing(b)=1; peak state ขึ้นกับจำนวน columns ไม่ใช่จำนวน rows. ทดสอบ blank string, whitespace และ malformed row แยกกัน.
20. ใช้ token set ที่ประกาศ เช่น true/false หลัง strip/lower; ตัดสิน bool ก่อน numeric เพราะ bool เป็น subclass ของ int ใน Python. ['true','False']→boolean, ['true','1']→mixed ถ้าไม่รองรับ 1/0; ทดสอบ missing,unknown token และบันทึกจำนวน invalid แทนบังคับแปลงทุกค่า.

## 02
1. Scalar มีหนึ่งค่า shape (), vector มีแกนเดียว เช่น (3,), matrix มีสองแกน เช่น (2,3). `[1,2,3]` กับ `[[1,2,3]]` มีสมาชิกสามตัวเหมือนกันแต่ shapes ต่างกัน; แกนที่เพิ่มมีผลต่อ broadcasting และ matrix multiplication.
2. สำหรับ vectors ยาว d เท่ากัน `x·w=Σ_j x_j w_j`. [1,2]·[2,-1]=2-2=0 เป็น orthogonality เมื่อทั้งสอง vector ไม่เป็นศูนย์; ตรวจ inner dimension ก่อนคูณ.
3. Derivative วัดอัตราเปลี่ยน output ต่อ input ณ จุดหนึ่ง ไม่ใช่ค่าเฉลี่ยตลอดกราฟ. f(x)=x² มี f′(2)=4 จึงประมาณ f(2+.01)≈4+.04=4.04; ค่าจริง4.0401 ต่าง .0001 จาก curvature.
4. Gradient รวม partial derivatives ของ scalar-valued function เป็น vector. f(x,y)=x²+3y² มี ∇f=(2x,6y); ที่ (1,2) ได้ (2,12) shape (2,). ก้าว −η∇f ลดค่าท้องถิ่นเมื่อ η เหมาะสม ไม่รับประกัน global optimum ทุกฟังก์ชัน.
5. `P(A|B)=P(A∩B)/P(B)` เมื่อ P(B)>0. ถ้า P(A∩B)=.12,P(B)=.3 ได้ .4; ไม่เท่ากับ P(B|A) โดยอัตโนมัติเพราะ denominator ต่างกัน.
6. Matrix product (m,k)@(k,n) ต้องให้แกนที่ถูก sum มีความยาวเท่ากัน. (3,4) กับ (5,2) มี 4≠5 จึงไม่มี dot product ราย row-column; transpose สุ่ม ๆ ไม่ใช่การแก้จนกว่าจะรู้ความหมายของข้อมูล.
7. แต่ละ output `(Ax)_i` เป็น dot ของ row i ของ A กับ x. A=[[1,2],[3,4]],x=[2,-1] ให้ [0,2]; output ยาวเท่าจำนวน rows ของ A และ x ต้องยาวเท่าจำนวน columns.
8. NN เป็น composition ของ operations. ถ้า z=wx,b=0,h=z²,L=h แล้ว dL/dw=(2z)x; x=2,w=3 ให้ z=6 และ gradient24. Backprop คูณ local derivatives ตามเส้นทางและบวกเมื่อมีหลายเส้นทาง.
9. Population variance หาร n ของ population ที่รู้ครบ; unbiased sample variance หาร n−1 เมื่อประมาณ population variance จาก IID sample ที่ใช้ sample mean. [1,3] มี squared deviationsรวม2 จึง population variance1, sample variance2; sample n<2 ต้อง reject.
10. Correlation บอก linear association ไม่ระบุ causal direction หรือ confounders. อุณหภูมิทำให้ยอดไอศกรีมและการจมน้ำเพิ่มพร้อมกันได้ โดยไอศกรีมไม่ใช่สาเหตุที่พิสูจน์แล้ว; ต้องมี causal assumptions/experimental design เพิ่ม.
11. `np.asarray(x,dtype=float)*a` คูณ scalar เดียวทุกพิกัด. a=3,x=[1,-2,0]→[3,-6,0], shapeคง (3,); ทดสอบ scalar0,negative และไม่ mutate input.
12. ตรวจ `A.shape==B.shape` ก่อน `A+B` ถ้าสัญญาคือ matrix addition shape เดียวกัน. [[1,2]]+[[3,4]]→[[4,6]]; (2,1) กับ (2,) ต้อง reject แม้ NumPy broadcast ได้ เพราะนั่นเป็น operation อีกแบบ.
13. `cos(x,y)=dot(x,y)/(norm(x)*norm(y))`. [1,0] กับ [1,1] ได้1/√2≈.70710678; zero vector ไม่มี cosine ตามนิยามจึง reject. ตรวจ scaling positive ของ vector ไม่เปลี่ยน cosine และ shapes ต้องตรง.
14. ใช้ covariance/std ที่มี ddof เดียวกัน: corr=cov(x,y)/(std(x)std(y)). x=[1,2,3],y=[2,4,6] ได้1; feature constant ต้อง reject/คืนสถานะ undefined แทนหารศูนย์หรืออ้างว่า correlation0 มีความหมายเดียวกัน.
15. f′(x)=3x²+2. ที่ x=2 ได้14; central difference `(f(x+h)-f(x-h))/(2h)` h=1e-5 ควรใกล้14 ภายใน1e-7. ทดสอบหลาย x รวม0 และติดลบ ไม่ใช้ forward difference เป็นสูตรเดียวกันโดยไม่ระบุ error order.
16. สร้าง C shape(m,n), loop i เป็น row ของ A,j เป็น column ของ B,k เป็น shared axis แล้ว `C[i,j]+=A[i,k]*B[k,j]`. ทดสอบ non-square (2,3)@(3,4)→(2,4) เพื่อจับโค้ดที่บังเอิญผ่านเฉพาะ square matrices.
17. เมื่อ h เล็กมาก f(x+h) กับ f(x-h) แทบเท่ากัน การลบเสียเลขนัยสำคัญและ roundoff หารด้วย h ถูกขยาย. Sweep h เช่น1e-2..1e-12 แล้วเทียบ analytic derivative; error มักลดก่อนเพิ่ม ไม่รับประกัน h ยิ่งเล็กยิ่งแม่น.
18. Sample covariance ใช้ denominator n−1 หลังหัก sample means. x=[1,3],y=[2,6] ให้ผลรวม cross deviations4 และ sample covariance4; population covariance2. เพิ่ม n=1 rejection และเทียบ np.cov(...,ddof=1).
19. 2×2 ใช้ ad−bc; [[1,2],[3,4]] ได้−2. 3×3 ขยาย cofactor แถวแรกพร้อมสลับเครื่องหมาย +−+; identityได้1, duplicated rowsได้0, สลับสอง rows เปลี่ยนเครื่องหมาย. เทียบ np.linalg.det ด้วย tolerance ไม่ตรวจ float equality ทุกกรณี.
20. ทำ perturb หนึ่งพิกัดต่อครั้งโดยคงพิกัดอื่น ใช้ central difference. f=x²+3y² ที่(1,2)ต้องได้[2,12]; inputต้นฉบับต้องเหมือนเดิมหลังฟังก์ชัน, ทดสอบ(0,0),(-1,.5)และหลาย h เพื่อจับ index/aliasing bug.

## 03
1. shape คือความยาวแต่ละแกน, ndim คือจำนวนแกน, dtype คือชนิดการเก็บสมาชิก. np.zeros((2,3),dtype=np.float32) มี shape(2,3),ndim2,itemsize4 รวม payload24bytes; ไม่รวม overhead ของ array object.
2. `*` คูณ elementwise ตาม broadcasting ส่วน `@` เป็น matrix product. A=[[1,2],[3,4]],I=identity: A*I ตัด off-diagonal เหลือ[[1,0],[0,4]] แต่ A@I=A. จึงไม่ควรเปลี่ยนเครื่องหมายเพื่อให้ shape error หายเฉย ๆ.
3. เปรียบเทียบแกนจากท้าย แต่ละคู่ต้องเท่ากันหรือมีตัวหนึ่งเป็น1; (2,3)+(3,)→(2,3). แต่ (2,1)+(2,)→(2,2) ซึ่งรันได้แต่ผิดถ้าต้องการ residual ต่อสอง samples.
4. DataFrame เป็นตารางมีชื่อ columns,index และ dtype ต่อ column. Operations อาจ align ด้วย labels ไม่ใช่ตำแหน่ง; Series index [10,20] บวกกับ [20,30] จับตามชื่อ20และสร้าง missing ที่10/30 จึงต้องตรวจ index ก่อน assignment.
5. Histogram แบ่งช่วงค่าต่อเนื่องเป็น bins แล้วนับความถี่ ช่วยดู skew,modes,outliers. จำนวน/ขอบ bins เปลี่ยนภาพได้; แสดง units และจำนวนข้อมูลที่ใช้ พร้อมจำนวน missing ที่ไม่ได้เข้า plot.
6. Inner dimension4ตรงกัน ผลคือ(100,3): แต่ละหนึ่งใน100 samples ได้3 outputs. คำนวณ row0 คูณแต่ละ column ของ weights เพื่อตรวจค่าควบคู่ shape และห้ามสับสนกับ elementwise product.
7. axis0 เป็นแกน rows จึงรวม100 samples แยกต่อ4 features เหลือ(4,). `keepdims=True` ให้(1,4) ซึ่งช่วย broadcast กลับเข้า(100,4); mean(axis=1) ให้(100,) เป็นคนละสถิติ.
8. Vectorization ย้าย inner loops ไป native array kernels ลด Python dispatch และใช้ memory locality. ไม่รับประกันเร็วหรือใช้ memory น้อยทุกโจทย์: pairwise broadcasting อาจสร้าง N×N temporary ใหญ่ ต้องวัด time/peak memory พร้อมกัน.
9. การ drop อาจเปลี่ยน distribution และทิ้งข้อมูลบางกลุ่มมากกว่าอื่น. รายงาน missing rate ต่อ column/group แยก target missing จาก feature missing แล้วเลือก imputation/drop บน train protocol ที่ประกาศ.
10. Correlation matrix แสดง association คู่ตัวแปร มองไม่เห็นการควบคุม confounders หรือ temporal causality. ตรวจ scatterplots,selection bias และเหตุผลของ data collection; feature correlation สูงไม่ได้พิสูจน์ว่าการแทรกแซง feature จะเปลี่ยน target.
11. fit μ=X.mean(axis=0),σ=X.std(axis=0); ใช้ denominator1สำหรับ σ=0 แล้วคืน (X−μ)/safeσ. [[1,2],[1,4]]→[[0,-1],[0,1]]; เก็บ μ/σ จาก train แล้วใช้ transform เดิมกับ held-out.
12. ตัวอย่าง `df.loc[df['score']>=50].copy()` เลือก rows ที่ผ่านเกณฑ์และคืนสำเนาตาม contract. scores[40,50,80] ต้องเหลือ50,80; กำหนด policy ของ NaN และทดสอบว่า boolean mask มี index ตรง DataFrame.
13. ข้อมูล group[A,A,B],score[1,3,10] ใช้ groupby('group')['score'].mean() ได้A=2,B=10. ระบุว่าข้าม NaN หรือไม่; ตรวจ group null และชื่อผลลัพธ์ก่อน merge กลับเพื่อไม่ duplicate rows.
14. ใช้ scatter โดย x=study_hours,y=score พร้อม labels/units และจำนวนจุด. รักษา row alignment; missing ของสองคอลัมน์ต้องกรองด้วย maskเดียวกัน. แยก correlation จากเหตุผลเชิง causality แม้กราฟดูชันขึ้น.
15. ใช้ `np.quantile(values,[.25,.5,.75],method='linear')` และระบุ method. [1,2,3,4] ได้[1.75,2.5,3.25]; [] ต้องรายงานไม่มีข้อมูล ไม่คืนค่าศูนย์ที่ดูเหมือนวัดแล้ว.
16. X ต้องการ weights4พิกัด แต่ wมี3. ตรวจ feature selection และ preprocessing ก่อนแก้ให้ w shape(4,); ทดสอบด้วย Xหนึ่งrowและ4weights คำนวณมือ ไม่เติมศูนย์โดยไม่รู้ว่า featureใดหาย.
17. หาก y_pred shape(B,1),y shape(B,) ผลลบเป็น(B,B). เลือก contract เช่นทั้งคู่(B,) แล้ว reshape อย่างตั้งใจและ assert equality ก่อนหาค่าloss; B=2 ต้องได้2 residuals ไม่ใช่4.
18. ใช้ pd.to_numeric(errors='coerce') พร้อม mask `original.notna() & converted.isna()` เพื่อแยก bad tokens จาก missingเดิม. ['1','bad',None] มี invalid1และmissingเดิม1; เก็บ audit countsก่อนเติม/ลบค่าที่แปลงไม่ได้.
19. สร้าง inputsหลายขนาดด้วยseedคงที่ วัดหลายซ้ำด้วยperf_counterและwarmup รายงานmedianพร้อมmemory. ตรวจ np.allclose ของผลloop/vectorizedก่อนเทียบtime; แยกเวลาallocationและการคำนวณเมื่อเกี่ยวกับข้อสรุป.
20. Report ต้องระบุ file fingerprint/schema,row counts,missing/duplicates,สถิติต่อcolumnและplotsที่อ่านหน่วยได้. ใช้ CSVใหม่ที่มี constant column,bad numeric token,missing และduplicate เพื่อทดสอบการแจ้งเตือน ไม่ประเมินความถูกต้องจากไฟล์สะอาดไฟล์เดียว.

## 04
1. Sample คือหน่วยตัวอย่างหนึ่งหน่วย,feature คือข้อมูลที่รู้ตอนทำนาย,target คือสิ่งที่ต้องทำนาย. งานพยากรณ์รายวัน sample=(ร้าน,วันที่),feature=ยอดอดีต,target=ยอดวันถัดไป; targetอนาคตห้ามหลุดกลับเข้า features.
2. Schema ระบุชื่อ/ชนิด/range/nullabilityและความหมายของข้อมูล. ตัวอย่าง age:int≥0,target∈{0,1}; CSVที่อ่านผ่านยังผิด schemaได้. แยก missing optional column ออกจาก required columnหายและรายงาน path/rowที่ผิด.
3. `z=(x−μ_train)/σ_train`. train[1,3]มีμ2,σ_population1 จึงz[-1,1]; held-out5ได้3โดยใช้stateเดิม. σ0ต้องมีpolicy เช่นsafe denominator1 ไม่หารศูนย์.
4. One-hot ใช้ nominal categories ที่ไม่มีลำดับเชิงปริมาณ. train['red','blue']สร้างสองcolumns; categoryใหม่ต้อง rejectหรือใช้unknown bucket/ignoreตามpolicyที่บันทึก ไม่fit vocabularyใหม่ในproductionจนcolumnorderเปลี่ยน.
5. Leakage คือการใช้ข้อมูลที่ไม่ควรรู้ ณ ขั้นฝึก/เลือกโมเดล เช่นscalerfitจากtest หรือfeatureสร้างจากlabelอนาคต. ตรวจ data lineageและเวลาที่แต่ละfeatureพร้อมใช้; scoreสูงผิดปกติเป็นสัญญาณ ไม่ใช่หลักฐานสรุปสาเหตุอย่างเดียว.
6. Splitก่อนfitเพื่อให้ μ/σ/median/category vocabularyมาจากtrainเท่านั้น. เปลี่ยน held-out outlier แล้ว stateของpreprocessorต้องไม่เปลี่ยน; ถ้าใช้CVต้องfitภายในแต่ละfold ไม่fitครั้งเดียวก่อนแบ่งfold.
7. IDอาจเป็นlabelระบุentity ไม่ใช่quantityที่ระยะห่างมีความหมาย. ID100กับ101ไม่ได้แปลว่าใกล้กว่าระหว่าง100กับ500เชิงพฤติกรรม; ตรวจentityleakageและใช้encodingที่เหมาะกับdeploymentgroups.
8. Medianใช้ลำดับจึงไวต่อextreme magnitudeน้อยกว่าmean. [1,2,100]mean34.33,median2; fitmedianจากtrainและเพิ่มmissingindicatorเมื่อmissingมีความหมาย. Robustไม่แปลว่าปลอดbiasทุกmissingness mechanism.
9. Randomshuffleทำให้ข้อมูลอนาคตเข้าฝึกก่อนพยากรณ์อดีตและอาจให้rollingfeaturesข้ามเวลา. ใช้time-orderedหรือwalk-forward splitและgapเมื่อlabelwindowsซ้อนกัน; testsตรวจmax(train_time)<min(val_time).
10. แถวเหมือนกันอาจเป็นretryduplicate eventหรือคนซื้อสินค้าซ้ำจริง. ใช้event_id/source_time/ingestionidตามcontractเพื่อdedup ไม่ใช้drop_duplicatesทุกcolumnsจนลบvalidrepeatและเปลี่ยนfrequency.
11. fitmin/maxต่อfeatureบนtrain, transform=(X−min)/(max−min)โดยsafe denominator1สำหรับconstant. train[2,4]→[0,1],held-out6→2ซึ่งถูกต้องสำหรับextrapolation; อย่าclipโดยไม่ประกาศpolicy.
12. นับค่าที่ไม่missingต่อcolumn เลือกmodeด้วยtie-breakที่deterministicแล้วเก็บstate. ['b',None,'a','b']fillเป็นb; all-missingต้องfallbackที่ประกาศหรือreject ไม่เลือกcategoryจากvalidation.
13. ตรวจfinite,typeและlower/upperboundsที่รวม/ไม่รวมขอบตามschema. age[0,120]ผ่านเมื่อboundsinclusive; -1,121,NaNต้องมีissueที่ระบุrow,column,rule โดยไม่แปลงเป็น0เงียบ ๆ.
14. ใช้`df.iloc[indices]`เมื่อsplitterคืนpositionalindices ไม่ใช่locที่อิงindexlabels. ทดสอบindexlabels[10,20,30]กับtrain_idx[0,2]ต้องได้rows10,30; ตรวจsplitsไม่ซ้ำและครอบคลุมตามcontract.
15. สร้างindicatorจากmissingmaskก่อนimputation เช่น[1,NaN,3]→indicator[0,1,0]. fitmedianแล้วเติมค่าคนละขั้น เก็บลำดับcolumns; missingindicatorของtargetไม่ควรเป็นfeatureที่deploymentไม่มี.
16. ย้ายscaler.fitเข้าtrain-onlypipelineหรือแต่ละCVfold. เปลี่ยนtestoutlierจาก10เป็น1000แล้วtrain-scaledvaluesต้องเท่าเดิม; retrainด้วยsplitเดิมและรายงานscoreใหม่แทนเก็บscoreที่มีleakage.
17. Policy1 rejectunknownพร้อมerrorที่อ่านได้; policy2 mapเข้าunknownbucketหรือall-zeroเมื่อhandle_unknown=ignore. ทดสอบshape/columnorderคงเดิมและแยกunknownจากmissing; เลือกpolicyจากความหมาย/ความเสี่ยงของงาน.
18. ตรวจว่าตัวอย่างของentityเดียวกันหรือwindowซ้อนกันข้ามsplitหรือไม่ ตรวจfeatureที่มีtargetในอนาคต. เปลี่ยนเป็นwalk-forwardพร้อมtraining-onlypreprocessingแล้วเทียบnaivebaseline; อย่าแก้ด้วยrandomseedที่ทำscoreดูต่ำลงเฉย ๆ.
19. แบ่งuniquegroupsก่อนแล้วmaprowsกลับ; assert intersectionของgroup setsว่างทุกคู่. groups[A,A,B,C]ห้ามให้Aอยู่ทั้งtrain/val; รายงานclassbalanceและจำนวนgroupsเพราะrowratioอาจไม่ตรงเป้าหมายเป๊ะ.
20. CLIอ่านschema/dataแล้วคืน0เมื่อผ่าน,nonzeroเมื่อfail พร้อมรายงานissuesแบบmachine-readable. ทดสอบrequiredcolumnหาย,typeผิด,out-of-rangeและไฟล์หาย; วัดว่ารายงานระบุแถวแรกที่ผิดและไม่บันทึกartifactว่าvalidหลังfail.

## 05
1. Parameterเรียนจากtrain เช่นw,b; hyperparameterเลือกprotocol เช่นdegree,learningrate,λ. การเลือกhyperparameterจากvalidationเป็นส่วนหนึ่งของmodelselectionจึงห้ามใช้testช่วยเลือกโดยไม่ถือว่าtestถูกใช้แล้ว.
2. Lossให้สัญญาณที่optimize เช่นBCE ขณะที่metricบอกผลงานตามงาน เช่นF1/latency. Objectiveอาจรวมloss+regularization; ลดtrainobjectiveไม่ได้รับประกันF1หรือproductionqualityดีขึ้น.
3. `MSE=Σ(ŷ_i−y_i)²/n`. y=[1,3],ŷ=[2,1]errors[1,-2],squares[1,4],MSE2.5. ตรวจshapeตรงและn>0; ค่าMSEมีหน่วยtarget².
4. Overfitคือจับรายละเอียด/noiseของtrainจนgeneralizationแย่: trainerrorต่ำแต่held-outerrorสูง. ต้องเปรียบเทียบด้วยsplitที่ถูกต้องและlearningcurves; trainerrorต่ำอย่างเดียวไม่บอกว่ามีoverfitแน่นอน.
5. Baselineเป็นคู่เทียบที่เรียบง่ายและถูกprotocol เช่นpredicttrainmean. ถ้าโมเดลซับซ้อนแพ้baselineบนheld-outให้ตรวจpipeline/varianceก่อนเพิ่มlayers; baselineเองต้องไม่fitจากtest.
6. R²=1−SSE/SST. ถ้าทำนายแย่กว่าการใช้meanของyในชุดที่ประเมินจะSSE>SSTจึงR²<0; เช่นy=[0,2],ŷ=[10,10]SSE164,SST2ให้−81. ไม่clipคะแนนเป็น0เพื่อซ่อนความผิด.
7. Training scoreวัดข้อมูลที่ใช้fit จึงมองไม่เห็นการจำnoiseหรือleakage. แยกvalidationเลือกconfigและtestยืนยันครั้งสุดท้าย; ตรวจdistribution/time/groupของsplitก่อนตีความgap.
8. Biasสูงคือสมมติฐานจำกัดจนพลาดโครงสร้าง; varianceสูงคือผลไวต่อtraining sample. ทดลองfitหลายseed/samplesด้วยtestprotocolเดิมแล้วดูการกระจายprediction; การเพิ่มcapacityอาจลดbiasแต่เพิ่มvarianceและไม่แก้distribution shift.
9. ถ้าเลือกconfigurationจากtestค่าที่รายงานจะoptimisticจากการเลือกซ้ำ. Freezeconfigหลังvalidationแล้วใช้testครั้งสุดท้าย; ถ้าใช้testไปแล้วต้องประกาศและใช้held-outใหม่สำหรับข้อสรุปยืนยัน.
10. K-foldให้แต่ละsampleเป็นvalidationหนึ่งรอบและสรุปความแปรปรวนข้ามfold ช่วยประเมินเมื่อข้อมูลจำกัด. ไม่แก้group/timeleakageโดยอัตโนมัติ; ต้องเลือกsplitterและfitpreprocessingใหม่ภายในfold.
11. RMSE=sqrt(mean(error²)); errors[1,-2]ให้sqrt2.5≈1.58113883และหน่วยเดียวกับtarget. ทดสอบexactprediction0,shapeผิดและemptyinputที่ต้องreject.
12. Fitmedianจากy_trainแล้วทำนายค่าคงที่นั้นทุกsample. train[1,2,100]ได้2 ซึ่งminimizeabsoluteerrorของconstantpredictor; ห้ามใช้medianของvalidation/testเพื่อทำbaselineให้ดูดีขึ้น.
13. ทำK-foldหลายรอบด้วยseedที่กำหนดแต่ละรอบ เก็บper-foldmetricsและindices. ตรวจว่าแต่ละรอบvalidationครอบคลุมทุกrowครั้งเดียว,train/valไม่ซ้ำ และreportdistributionแทนเลือกseedที่ดีที่สุด.
14. f′(x)=2(x−3); x′=x−2η(x−3). เริ่มx0=0,η=.1ได้x1=.6,f1=5.76จากf0=9. Plotตำแหน่งบนcurveและlossต่อupdateด้วยparametersจากhistoryจริง.
15. Errorต่อoptimumคูณ(1−2η)ทุกก้าว. η=.01หดช้า,.1หดเร็ว,.9สลับข้างแต่ยังหด,1.1ขยายเพราะ|1−2η|=1.2. ใช้x0/จำนวนstepsเดียวกันและแสดงค่าที่divergeจริง.
16. Preprocessorเห็นvalidationrowsก่อนfitmodelจึงleak. ใส่scalerในPipelineแล้วfitเฉพาะtrain_idxแต่ละfold; testโดยแก้validationoutlierแล้วμ_trainต้องไม่เปลี่ยน พร้อมเก็บผลCVเดิมว่าไม่valid.
17. ตรวจcapacity,regularization,n_samples,labelnoiseและleakageก่อนลดcomplexity. ใช้learningcurveและหลายtrainfractionsที่splitเดิม; รายงานทั้งtrain/valไม่เลือกเฉพาะepochที่ดูสวยโดยใช้test.
18. ความต่างรวมทั้งผลของโมเดลและความยากของsplit จึงสรุปเหตุไม่ได้. Freezeindicesและseedคู่เทียบก่อนเปลี่ยนโมเดล; ถ้าต้องหลายsplitให้ใช้pairedcomparisonบนsplitsชุดเดียวกัน.
19. คงvalidationเดิม แล้วsubsampletrain10..100%ด้วยseedที่ประกาศ fitpipelineใหม่แต่ละครั้ง. Plotจำนวนtrainingexamplesกับtrain/valerrorพร้อมvarianceหลายซ้ำ; ห้ามเพิ่มvalidationกลับเข้าtrainตอนfractionสูง.
20. Evaluatorสร้างindicesแล้วเรียกfit_predictต่อfold ตรวจreturnedpredictionsยาวเท่าval_idxและfinite. Testcallbackต้องasserttrain/valdisjoint; เก็บmetricsต่อfoldและaggregateแบบระบุweight ไม่แอบaverageprobabilitiesข้ามคนละsamples.

## 07
1. Sigmoid σ(z)=1/(1+exp(−z))แปลงlogitจริงเป็น(0,1); z0→.5,zlog3→.75. ใช้สูตรstableเมื่อzติดลบเพื่อหลีกoverflow และความน่าจะเป็นนี้ยังต้องตรวจcalibrationบนheld-out.
2. Logitคือlog-odds log(p/(1−p)); p=.8ให้log4≈1.38629. Modelscorez=Xw+bเป็นlogitก่อนsigmoid ไม่ใช่classlabelหรือpercentage.
3. BCE=−[ylogp+(1−y)log(1−p)]เฉลี่ยsamples. y1,p.8ให้−log.8≈.22314355; y0,p.8ให้1.60943791. ใช้logit-formsoftplus(z)−yzเพื่อความstable.
4. Precision=TP/(TP+FP)ในpredictedpositives. TP8,FP2ให้.8; denominator0ต้องประกาศpolicyเช่นzero_division=0พร้อมแจ้งว่าไม่มีpositiveprediction ไม่ตีความว่าprecisionที่วัดแล้วดี.
5. Recall=TP/(TP+FN)ในactualpositives. TP8,FN12ให้.4; FNมีบทบาทต่างจากFPจึงต้องอธิบายต้นทุนงานก่อนเลือกthreshold.
6. แยกσ(z)=(1+e^(−z))^(−1)แล้วchainruleได้e^(−z)/(1+e^(−z))²=σ(1−σ). ที่z0derivative.25; saturatedlogitให้localderivativeเล็กแต่BCEgradientต่อlogitคือp−y.
7. Bernoullilikelihood=p^y(1−p)^(1−y). loglikelihood=ylogp+(1−y)log(1−p); minimizenegativeและเฉลี่ยเป็นBCE. Chainกับsigmoidให้dL/dz=p−yจึงใช้Xᵀ(p−y)/nสำหรับw.
8. .5เหมาะกับสมมติฐานบางชุดของcostsและcalibrationเท่านั้น. หากFNแพงอาจเลือกthresholdต่ำลงจากvalidation; กำหนดoperatingpointตามrecall/precision/costแล้วประเมินtestด้วยthresholdที่freeze.
9. ข้อมูล99%negative โมเดลทำนายnegativeทั้งหมดได้accuracy99%แต่positive recall0. แสดงconfusionmatrix,PRcurveและmetricsแยกclass; อย่าใช้accuracyอย่างเดียวเลือกโมเดล.
10. ลดthresholdมักเพิ่มTPและFPทำให้recallสูงขึ้นแต่precisionอาจลด; ไม่รับประกันmonotonicprecisionทุกthresholdจากsamplefinite. Sweepโดยรวมtiesและคงprobabilitiesเดิมเพื่อแยกผลdecisionruleจากtraining.
11. Specificity=TN/(TN+FP). TN18,FP2ให้.9; testallnegativepredictionsและไม่มีactualnegativesโดยใช้นโยบายdenominator0ที่ประกาศ.
12. Binarybalancedaccuracy=(TPR+TNR)/2. TP8,FN2,TN45,FP45ให้(.8+.5)/2=.65แม้rawaccuracy.53. ทดสอบall-oneclassกรณีsupportไม่ครบและแจ้งข้อจำกัด.
13. เรียงuniqueprobabilitiesแล้วคำนวณconfusionmatrixทุกthresholdด้วยกฎp≥τเดียวกัน. p=[.2,.8],y=[0,1]ที่τ.5ได้TP1,TN1; เพิ่มจุดendpointเพื่อให้เห็นpredict-none/all.
14. ใช้thresholdบนแกนxและprecision/recallบนแกนy พร้อมdenominator0policy. เก็บthresholdที่เลือกจากvalidationเท่านั้น; ห้ามfitโมเดลใหม่ทุกthresholdเพราะจะเปลี่ยนทั้งสองปัจจัยพร้อมกัน.
15. ใช้J=BCE+λΣw²ไม่ลงโทษbiasตามนิยาม แล้วgw=Xᵀ(p−y)/n+2λw. คงsplit/scalerเดิมsweepλและดูvalidationBCE/calibrationกับcoefnorm; ไม่สรุปว่าλเพิ่มแล้วaccuracyต้องดีขึ้น.
16. log(sigmoid(z))ตรง ๆ อาจได้log0หรือexp overflow. ใช้`np.logaddexp(0,z)-y*z`; z=1000,y1ให้loss≈0และz=−1000,y1ให้1000ซึ่งfinite. ตรวจinput/gradientก่อนupdateแทนแค่clipกราฟ.
17. ตรวจTP,FNและsupportของminority ตรวจthreshold/calibration/weightingแล้วเลือกระดับrecallตามvalidationcost. accuracy97%ไม่ปิดปัญหาrecall5%; เปรียบเทียบกับmajoritybaselineและรายงานfalse positivesที่เพิ่ม.
18. ตรวจdistribution shift,prior shift,leakageและreliabilitycurveแยกslice. Calibrateบนheld-outcalibrationsetโดยคงtestไว้; temperature/isotonicไม่แก้ข้อมูลนอกdistributionทุกแบบและต้องวัดหลังdeploy.
19. เรียงscoresลดลงรวมtiegroup จากthreshold∞เริ่ม(0,0)แล้วสะสมTP/FPหารactualpositive/negative. คำนวณtrapezoidAUCและtestperfectranking1,reversed0,tiedscores.5; ไม่มีclassใดclassหนึ่งให้undefined/error.
20. ฝึกbinaryclassifierหนึ่งตัวต่อclassด้วยtargets1[y==k]และdata/scalerprotocolเดียวกัน. Predictionargmaxscoresต้องกำหนดcalibrationpolicy; OvRsigmoidsไม่รวมเป็น1เอง ห้ามเรียกprobabilitydistributionโดยไม่normalize/ประกาศความหมาย.

## 08
1. Euclidean `d(x,z)=√Σ_j(x_j−z_j)²`. [0,0]กับ[3,4]ได้5; squaredL2ให้25และจัดอันดับเหมือนกันสำหรับnonnegativefinitevalues แต่ไม่เท่ากันเมื่อเอาไปใช้1/d weighting.
2. Manhattan `Σ|x_j−z_j|`; จุด[0,0]กับ[3,4]ได้7. Geometryของunitballต่างจากL2จึงอาจเปลี่ยนneighborordering ไม่ใช่แค่คูณทุกdistanceด้วยconstant.
3. kเล็กทำboundaryยืดหยุ่นไวต่อnoise จึงมักlowbias/highvariance; kใหญ่smoothมากขึ้นแต่ปิดกลุ่มเล็กได้. เลือกจากvalidationด้วยscalerในfoldไม่ใช้trainaccuracyที่k1เกือบจำข้อมูล.
4. Distanceรวมหน่วยทุกfeature. incomeต่าง10000อาจกลบageต่าง1หากไม่scale; fittransformบนtrainแล้วใช้stateเดิมกับquery. Standardizationไม่จำเป็นว่าคือmetricที่เหมาะทุกงาน ต้องดูunitsและdomain.
5. KNNแบบพื้นฐานเก็บtrainvectors/labelsและอาจสร้างindex ไม่มีการfitcoefficientเหมือนlinearregression. เวลาและmemoryส่วนใหญ่ย้ายไปprediction/search; preprocessingยังมีfittedstateที่ต้องไม่leak.
6. Irrelevantdimensionเพิ่มnoiseในdistanceทำให้จุดใกล้ตามfeatureที่เกี่ยวข้องหลุดอันดับ. ทดลองappendrandomfeatureด้วยseedเดิมและscaleก่อนเทียบ; ทดสอบmetric/featureselectionด้วยvalidationแทนปรับบนtest.
7. เมื่อdimensionสูง ปริมาตรกระจายและระยะneighborอาจใกล้เคียงกัน ทำให้localityและtreepruningอ่อนลง. วัดnearest/farthestdistancecontrastกับlatencyและquality ไม่ใช่สรุปdimensionthresholdเดียวใช้ได้ทุกdataset.
8. Uniformให้สมาชิกkตัวน้ำหนักเท่ากัน ส่วนdistanceweightให้จุดใกล้มีน้ำหนักมาก เช่นweights1/max(d,ε). ถ้าqueryตรงจุดtrainให้ใช้เฉพาะzero-distancepointsและaggregateด้วยtiepolicyแทนให้εบิดผล.
9. หนึ่งquerybruteforceใช้O(Nd)คำนวณdistances บวกtop-kselectionตามalgorithm; Qqueriesใช้O(QNd). การสร้างtensor(Q,N,d)อาจกินmemoryสูง จึงต้องแยกcomplexityเวลาจากpeaktemporarymemory.
10. KDTree/BallTreeอาศัยขอบเขตdistanceเพื่อตัดcandidate; high-dimensionaldataมักตัดได้น้อยจนเกือบscanทั้งหมดพร้อมindexoverhead. Benchmarkบนdistributionจริงและเทียบbruteforcevectorizedก่อนเลือก.
11. สำหรับX(Q,d),Z(N,d)ใช้`sum((X[:,None,:]-Z[None,:,:])**2,axis=-1)`ได้(Q,N)แล้วsqrt. X=[[0,0]],Z=[[3,4],[0,0]]→[[5,0]]; validateinnerdimsและใช้chunkเมื่อQNdใหญ่.
12. เลือกkneighborแล้วรวมvotesต่อclassตามclassorderที่fitไว้ หารtotalweightแต่ละquery. labels[0,1,1],uniformให้[1/3,2/3]; ทดสอบclassesที่ไม่ปรากฏในneighborให้prob0และrowsum1.
13. แทนclassvoteด้วยmeanหรือweightedmeanของy_neighbors. y=[2,4],weights[1,3]ให้3.5; zero-distancepolicyเหมือนclassifier และผลไม่จำเป็นเป็นlabelที่เคยเห็น.
14. ใช้odd/evenk1..31ที่k≤n_train พร้อมvalidation/CVคงเดิมandscalerในfold. รายงานqualityและpredictionlatency ไม่เลือกจากtrainaccuracy; tiesต้องใช้policyคงที่ทุกk.
15. เปลี่ยนเฉพาะMinkowskip=1กับ2ในpipelineเดียวกัน. สร้างจุดquery[0,0],candidates[3,0],[2,2]: L1ให้3กับ4จึงเลือกแรก,L2ให้3กับ√8จึงเลือกหลัง แสดงdistanceที่ทำให้อันดับเปลี่ยน.
16. ตรวจstd/rangeของincomeเทียบfeaturesเดิมและdistancecontributionsรายfeature. FitStandardScalerบนtrainหรือเลือกdomainmetric แล้วเทียบsplitเดิม; อย่าลบfeatureเพียงเพราะหน่วยใหญ่โดยไม่ตรวจว่ามีsignal.
17. หารด้วยd+εทุกจุดอาจให้nonzero-distanceจำนวนมากกลบexactmatch. ถ้ามีd0ให้aggregateเฉพาะจุดที่d0และกำหนดtiepolicyเมื่อlabelsขัดกัน; testqueryตรงsampleหนึ่งและduplicatevectorsต่างclass.
18. Bruteforceโตประมาณ100เท่าในNเมื่อปัจจัยอื่นคงที่. Profiledistance/topkและbatching; พิจารณาindex/approxsearchพร้อมวัดrecallและlatencyจริง ไม่รายงานindexbuildtimeแทนquerytime.
19. แบ่งqueriesเป็นchunksขนาดBแล้วคำนวณ(B,N)distancesและคืนtopk ก่อนต่อผลลัพธ์. ทดสอบเทียบunchunkedหลายBรวมB1และlastpartialchunk; peakmemoryลดแต่ยังO(QNd)totalcompute.
20. สร้างmeshgridของสองfeatures flattenเป็น(M,2),predictแล้วreshapeกลับgrid. Overlaytrainpoints,แสดงk/scaling/metric; boundsต้องอยู่ในช่วงที่ประกาศและอย่าfitใหม่บนgridเพื่อให้boundaryดูดี.

## 09
1. Bayes `P(c|e)=P(e|c)P(c)/Σ_k P(e|k)P(k)` เมื่อevidenceprobability>0. prior.4,like1.8,like0.2ได้.32/(.32+.12)=.7272727; denominatornormalizeทุกclass.
2. Priorคือความเชื่อ/ความถี่classก่อนเห็นfeaturesของsampleนั้น. Empiricalprior=n_c/n_trainต้องfitจากtrain; เปลี่ยนpriorโดยคงlikelihoodทำให้posteriorเปลี่ยน จึงไม่ใช่probabilityofevidence.
3. Likelihoodมองfeaturesภายใต้classที่กำหนด `P(x|c)`. เช่นP(word='offer'|spam)=.8ไม่ใช่P(spam|offer); ต้องรวมpriorกับevidenceอีกclassesก่อนสรุปposterior.
4. Naiveassumptionคือfeaturesconditionallyindependentgivenclass จึงP(x|c)=Π_jP(x_j|c). ไม่ได้สมมติfeaturesindependentโดยไม่conditionclass; duplicatefeatureอาจนับหลักฐานเดิมซ้ำจนมั่นใจเกินจริง.
5. Multinomialsmoothing`P(w|c)=(count_w,c+α)/(total_tokens_c+αV)`. count0,total5,V3,α1ได้1/8; ต้องปรับdenominatorด้วย ไม่ใช่แทนแค่zeroด้วยเลขเล็ก.
6. Productsของsmallprobabilitiesอาจunderflowเป็น0 ขณะที่logเปลี่ยนproductเป็นsum. ใช้logprior+Σloglikelihoodแล้วnormalizeด้วยlogsumexp; ไม่แปลงกลับexpก่อนnormalizeจนunderflowซ้ำ.
7. GaussianNBประมาณแต่ละcontinuousfeatureด้วยGaussianที่มีμ_cj,σ²_cjและconditionalindependence. แต่ละfeatureมีGaussianของตัวเองในแต่ละclass ไม่ได้แปลว่าjointdistributionทั้งหมดทุกclassเป็นGaussianเดียว.
8. MultinomialNBเหมาะกับnonnegativecounts/weightsเช่นbag-of-words. Negativecenteredfeaturesผิดdomain; TF-IDFอาจใช้ได้เป็นengineeringchoiceแต่ไม่ใช่integercountgenerativemodelตรงตามสมมติฐานทุกประการ.
9. BernoulliNBมองfeaturepresence0/1และรวมlogP(x=0|c)สำหรับabsenceด้วย ส่วนMultinomialNBนับtokenfrequency. เอกสารมีคำเดิม10ครั้งให้effectต่างกัน; ต้องfitbinarizationpolicyจากtrain.
10. Correlatedfeaturesถูกนับเป็นหลักฐานอิสระซ้ำจนlogoddsใหญ่เกินจริง. เปรียบเทียบcalibration/Brier/loglossก่อนหลังduplicatefeature; accuracyอาจคงเดิมแต่ความมั่นใจผิดมากขึ้น.
11. คำนวณjointlogscore Gaussianแต่ละclassแล้วลบlogsumexpต่อrowเพื่อได้normalizedlogposterior. Testexp(result).sum(axis=1)=1,shape(N,C),extremeinputfiniteและclassorderไม่เปลี่ยนจากfit.
12. ให้sเป็นlogscores,m=max(s); logZ=m+log(sum(exp(s−m))). probabilities=exp(s−logZ). s=[1000,1000]ต้องได้[.5,.5]แทนoverflow; emptyclassaxisต้องreject.
13. Fitθ_cj=(positive_count_cj+α)/(n_c+2α); score=logprior+Σ[x_jlogθ+(1−x_j)log(1−θ)]. Datasetclass0=[0,0],class1=[1,1]กับα1ต้องให้query[1,1]เอนclass1และquery[0,0]เอนclass0.
14. Sweepαบนvalidationโดยคงvocabulary/splitและรายงานloglossร่วมaccuracy. αเล็กไวต่อrare/unseenwords,αใหญ่ดึงlikelihoodเข้าหาuniform; ไม่เลือกαจากtestหรือเพิ่มvocabularyจากvalidation.
15. เปรียบเทียบprior_empiricalกับprior_fixedที่sum1และpositiveโดยคงclasslikelihoodเดิม. สร้างimbalancedtrainแต่balancedvalidationเพื่อดูthreshold/calibration; fixedpriorเป็นassumptionที่ต้องระบุ ไม่ใช่วิธีแก้imbalanceที่รับประกัน.
16. ย้ายcomputationไปlogspaceและnormalizeหลังลบmax. TestΠ(1e-3)500ตัวที่productเป็น0แต่sumlogsfinite; ผลclassrankingต้องตรงกับhigh-precision/smallcaseที่ยังคูณตรงได้.
17. Variance0ทำให้Gaussianlogdensityหารศูนย์. เติมvariancefloorตามscaleที่ประกาศfitจากtrainแล้วทดสอบconstantfeature; ต้องตรวจว่าfeatureconstantเป็นdataqualityissueหรือจริง ไม่เติมnoiseจากtestเพื่อกลบ.
18. StandardScalerสร้างnegativevaluesซึ่งขัดMultinomialNBnonnegativeinput. ใช้counts/TF-IDFที่nonnegativeหรือเลือกGaussian/Bernoullimodelที่เหมาะ; testsต้องrejectnegativecountsก่อนfitพร้อมตำแหน่งfeature.
19. Tokenizeด้วยกฎcase/punctuationที่versioned,fitvocabularyบนtrain,นับcountsแล้วfitclasspriors/likelihoodwithsmoothing. queryที่มีแต่OOVควรได้priorเมื่อignoreOOV; testroundtripnormalizationและvocabไม่เปลี่ยนหลังtransform.
20. Duplicatefeatureที่สัมพันธ์เต็มที่แล้วfitNBใหม่บนsplitเดิม เทียบaccuracy,loglossและcalibration. Evidenceซ้ำทำlogodds contributionเพิ่มโดยไม่ได้เพิ่มข้อมูลอิสระ; ต้องรายงานผลที่วัด ไม่บังคับว่าaccuracyจะแย่ทุกseed.

## 10
1. Gini=`1−Σ_c p_c²`. labeldistribution[.75,.25]ให้1−.5625−.0625=.375; pureleafให้0. ตรวจcountsnormalizeจากnodeนั้น ไม่ใช้classpriorทั้งdataset.
2. Entropy`−Σp_c log2 p_c`โดยให้0log0=0. balancedbinaryได้1bit,pureได้0; baseของlogต้องตรงตอนเทียบgain ไม่ผสมnatsกับbits.
3. Gain=impurity_parent−Σ_child(n_child/n_parent)*impurity_child. ลูกpureหนึ่งตัวไม่ได้พอ ต้องถ่วงขนาดทั้งสองลูก; หากweightedchildimpurityเท่าพ่อgain0และไม่มีประโยชน์ตามobjectiveนี้.
4. Leafเป็นnodeปลายทางที่ไม่splitต่อ คืนclassdistributionหรือค่าทำนายคงที่. Leafcounts[3,1]ให้Pclass1=.25และmajorityclass0; ไม่ใช่ความน่าจะเป็นที่calibratedแน่นอนบนข้อมูลใหม่.
5. max_depthจำกัดจำนวนการsplitตามทางroot→leaf; นิยามrootdepth0ให้stumpdepth1. เพิ่มdepthเพิ่มcapacityแต่ไม่รับประกันvalidationดีขึ้น; testsต้องตรวจจริงว่าnodesไม่เกินเงื่อนไข.
6. labels[0,0,0,1]มีp0=.75,p1=.25จึงGini=.375. ถ้าแบ่งเป็น[0,0]กับ[0,1],weightedGini=.25,gain=.125; แสดงการถ่วง2/4ต่อchildเพื่อจับสูตรที่ใช้averageผิด.
7. Treeเลือกthresholdตามลำดับค่าต่อfeature จึงinvariantต่อstrictlymonotonicpositiveaffinetransformเมื่อadjustthresholdตาม. KNNใช้distanceข้ามfeaturesจึงscaleมีผล; caveatเรื่องtiesและfloating-pointยังต้องตรวจ.
8. Deep treeสร้างregionsเล็กจนleafอาจมีsampleเดียวจึงจำlabelnoise. ใช้minleaf/depth/pruningและvalidation; trainaccuracy100%อาจเหมาะข้อมูลไร้noiseหรือoverfitก็ได้ ต้องดูheld-out.
9. Greedyเลือกsplitที่gainดีที่สุดตอนนี้โดยไม่ค้นทุกsubtreecombination. Splitที่gainน้อยตอนต้นอาจนำไปtreeรวมดีกว่า; จึงไม่พิสูจน์globaloptimumจากการที่แต่ละnodeหาbestlocalได้.
10. MDIสะสมweightedimpuritydecreaseจึงเอนหาfeaturesที่มีcandidatevaluesมาก และแบ่งimportanceระหว่างcorrelatedfeaturesไม่เสถียร. เทียบheld-outpermutationimportanceและตรวจleakage ไม่ตีความเป็นcausalimportance.
11. `score(X,y)=mean(predict(X)==y)`หลังตรวจNเท่ากันและnonempty. Exactlabels[0,1]ได้1,pred[0,0]ได้.5; scoreไม่ควรเรียกfitหรือใช้yในการเลือกbranch.
12. เก็บn_samplesและclasscountsในnodeตอนfitเพื่อใช้weightedgain/debug. rootcountต้องเท่าผลรวมchildcountsทุกsplit; testleafที่stopเพราะpure/minleafยังมีcountsถูกต้อง.
13. เลือกbestvalidsplitแล้วหยุดถ้าgain<min_impurity_decreaseตามinequalityที่ระบุ. Datasetที่bestgain.125กับthreshold.2ต้องไม่split; threshold0ยังต้องป้องกันnonpositivegain/emptychildrenตามcontract.
14. Pretty-printerเดินtreeแบบpreorder แสดงdepth,feature,threshold,counts,impurityและleafprediction. Testsmalltreeที่thresholdรู้ด้วยมือและจัดลำดับleft/rightชัด; อย่าให้printermutatetreestate.
15. สร้าง2Dmeshแล้วpredictด้วยtreeที่fitไว้ครั้งเดียว ระบายclassพร้อมoverlaytrain/validation. แสดงmaxdepth/minleafและunits; รูปboundaryแบบaxis-alignedเป็นผลจากsplitcriterion ไม่ใช่เส้นsmoothที่interpolateเอาเอง.
16. ตรวจbasecases purelabels,maxdepth,min_samples และต้องมีvalidsplitที่แบ่งsamplesจริง. ถ้าthresholdเท่าค่าสุดท้ายทำให้subsetไม่เล็กลง recursionจะวน; testallfeaturesconstantต้องคืนleafทันที.
17. ตัดcandidateที่n_leftหรือn_rightเป็น0หรือไม่ถึงminleafก่อนคำนวณgain. เช่นxทุกตัวเท่ากันไม่มีsplitที่valid; อย่าหาร0แล้วเลือกNaNgainเพราะsortbehavior.
18. ประเมินdepth/minleaf/pruningจากvalidationที่ไม่ใช้fit และตรวจduplicates/groupsleakage. เก็บtrain/validationlearningcurvesกับleafsizes; ไม่รายงานtrain100%แทนความสามารถgeneralize.
19. Regressionleafคืนmean(y_node),splitลดweightedMSEหรือSSEด้วยreductionเดียวกัน. y=[1,1,5,5]แยกสองคู่ให้SSE0จากparentSSE16; testconstanttargetและminleafเพื่อกันoverfittingsampleเดียว.
20. เปรียบเทียบcost `R(T)+α|leaves|` โดยRใช้trainingriskนิยามเดียวกัน; collapse subtreeเมื่อcostleaf≤costsubtree. αสูงควรไม่ทำจำนวนleavesเพิ่มตามpruningpath; เลือกαด้วยvalidationและทดสอบpredictionหลังcollapse.

## 11
1. Bootstrapสุ่มnindicesแบบคืนกลับจากntrainrows. indices[0,0,2]จากn3ทำsample0ซ้ำและไม่เห็นsample1; เก็บindicesเพื่อหาOOBและทำซ้ำด้วยseed ไม่ใช่shuffleโดยไม่คืนกลับ.
2. Baggingfitสมาชิกบนbootstrapdatasetsแล้วaggregateเพื่อลดvarianceเมื่อสมาชิกไม่ได้เหมือนกันทั้งหมด. Parallelfitได้เพราะแต่ละmemberไม่รอresidualของตัวก่อน ต่างจากboostingที่สร้างตามลำดับ.
3. max_featuresกำหนดfeaturecandidatesที่แต่ละsplitพิจารณา ช่วยdecorrelatetrees. ต้องสุ่มใหม่ตามpolicyและseedของmember ไม่ใช่ทิ้งfeaturesจากdatasetถาวรทุกtreeโดยไม่ตั้งใจ.
4. OOBของtreeคือtrainingrowsที่ไม่ถูกbootstrapของtreeนั้น. ประเมินsampleหนึ่งจากเฉพาะtreesที่sampleนั้นเป็นOOB; ถ้าไม่มีOOBvotesต้องรายงานว่าnotcovered ไม่ใช้in-bagpredictionแทนเงียบ ๆ.
5. จัดclasscolumnsให้ตรงglobalclassesแล้วaverageprobabilitiesต่อsample ก่อนargmax. สองtrees[.9,.1]กับ[.2,.8]เฉลี่ย[.55,.45]; averagingclassIDsให้.5ไม่ใช่classifierที่มีความหมาย.
6. Varianceของaverageขึ้นกับcovariancesระหว่างtrees; ถ้าทุกtreeผิดเหมือนกันการเพิ่มจำนวนไม่ช่วยมาก. Bootstrapกับfeaturesubsamplingเพิ่มdiversity แล้ววัดvalidation/OOB ไม่สมมติว่าindependenceสมบูรณ์.
7. โอกาสไม่ถูกเลือกในหนึ่งdrawคือ1−1/n; nindependentdrawsจึง(1−1/n)^n→exp(−1)≈.367879. นี่เป็นexpectedfractionOOBต่อtree ไม่ใช่รับประกันว่าแต่ละtreeมีOOB36.8%เป๊ะ.
8. การเพิ่มtreesมักลดMonteCarlo/ensemblevarianceแล้วplateau แต่biasของbaselearnerไม่ได้หายเอง. คงdepth/maxfeaturesและseedstreamเดิม วัดquality,latency,memory; อย่าตีความtrainscoreอย่างเดียว.
9. OOBสุ่มจากช่วงเวลาเดียวกับtrainingและอาจมีอนาคต/neighborwindowsของsampleนั้นในtree. ใช้walk-forwardvalidationสำหรับforecastdeployment; OOBยังใช้diagnosticsภายใต้assumptionsที่ระบุแต่ไม่แทนเวลาจริง.
10. ExtraTreesสุ่มthresholdsมากขึ้นแทนค้นbestthresholdเต็มชุดและค่าdefaultbootstrapอาจต่างกันตามAPI. เมื่อเปรียบเทียบให้ตั้งbootstrap/maxfeatures/depthชัดและวัดbias/variance ไม่กล่าวว่าExtraTreesคือRFที่เพิ่มจำนวนต้นเฉย ๆ.
11. เก็บOOBprobabilitiesจากสมาชิกที่sampleนั้นไม่เคยเห็น แล้วสร้างconfusionmatrixเฉพาะsamplesที่มีvotes. Testglobalclassorderและcoveragecount; ถ้าn_estimatorsน้อยบางsamplesอาจไม่coveredต้องแยกรายงาน.
12. `max_features=None`ให้พิจารณาทุกfeatureในแต่ละsplitตามcontract. Testdatasetd3ว่าcandidatepoolมี3featuresจริงและstillbootstraprows; ไม่ตีความNoneเป็น0featuresหรือข้ามfit.
13. รวม `(n_node/n_total)*gain_node` ต่อfeatureแล้วnormalizeเมื่อtotalgain>0. ถ้าallconstantfeaturesให้importanceเป็นzerosพร้อมnote; testimportance.sum≈1เมื่อมีusefulsplitและไม่เป็นcausalclaim.
14. วัด1..200treesบนsplit/hyperparamsเดิม เก็บfit/querytimeและpeakmemoryพร้อมqualityหลายseed. สมาชิกของNtreesควรเป็นprefixเดียวกับN+1เพื่อให้curveแยกผลจำนวนtreeจากการเปลี่ยนrandomstream.
15. PlotcoveredOOBscoreกับn_estimatorsพร้อมOOBcoverageและuncertainty. ไม่เติมuncoveredrowsด้วยtraininglabels; scoreอาจแกว่งไม่monotonicเพราะfinitevotesและsamplesที่ถูกประเมินเปลี่ยน.
16. การสร้างRNGใหม่ด้วยseedเดิมภายในloopทำให้bootstrapและfeaturechoicesเหมือนกัน. ใช้parentRNGสร้างmemberseedsต่างกันแต่reproducible; testsampleindexlistsไม่เหมือนกันทุกต้นและrerunseedเดิมได้เหมือนเดิม.
17. ตรวจmemberclassalignmentและnormalizeweightsด้วยจำนวนtreesจริง. Treeที่bootstrapขาดclassหนึ่งยังต้องส่งprob0ในcolumnของclassนั้น; testimbalancedtinydataและrowsum1หลังaggregate.
18. Fitparallelเร็วไม่ได้ทำqueryผ่านtreesน้อยลง. Profileจำนวนtree/depth/batchingและcachelocality; วัดp50/p95บนquerysizeจริง แล้วลดcapacityหรือcompressโดยตรวจqualityซ้ำ.
19. RFregressorbootstrapแล้วfitregressiontreesและaveragefloatpredictions. Targets[2,4]กับสมาชิกที่ทำนาย2และ4ต้องเฉลี่ย3; OOBaggregationต้องนับcoverageแยก และvalidateMSEด้วยsplitเดิม.
20. วัดbaselineheld-outmetricแล้วshuffleหนึ่งfeatureในheld-outเท่านั้นหลายซ้ำ คำนวณscoredecreaseตามทิศmetric. คงmodelไม่retrain; correlatedfeaturesอาจทดแทนกันจึงimportanceเล็กได้ทั้งที่กลุ่มfeaturesมีประโยชน์.

## 12
1. Baggingfitสมาชิกอิสระแล้วaverage ส่วนboostingเพิ่มผู้แก้errors/negativegradientsตามลำดับ. Boostingมีcapacityโตตามroundsและηจึงต้องearlystopจากvalidation; ไม่ใช่แค่baggingที่ใช้treesมากขึ้น.
2. Weightederror=Σ_i w_i 1[h(x_i)≠y_i]/Σ_i w_i. Weights[.1,.2,.7]ผิดเฉพาะตัวแรกได้ε.1 ไม่ใช่1/3; classifierbinaryใช้labels±1ให้สูตรupdateตรงกัน.
3. BinaryAdaBoost `α=.5 log((1−ε)/ε)`; ε.2ให้.693147. ε0หรือ1ต้องมีnumerical/stoppingpolicy; ε≥.5ไม่เป็นweaklearnerที่มีpositivevoteตามทิศเดิม.
4. Gradientboostingfitnegativegradientของobjectiveที่currentensembleคำนวณได้. สำหรับhalf-squaredlossคือy−F; classifierlogisticใช้y−sigmoid(F)ในlogitspace จึงห้ามยืมrawresidualสูตรเดียวไปทุกloss.
5. ηshrinkscorrection `F_new=F_old+ηh`. ค่าเล็กทำให้แต่ละก้าวระวังขึ้นแต่ต้องพิจารณาร่วมrounds/depth; การลดηโดยคงroundsต่ำอาจunderfit ไม่รับประกันqualityดีเสมอ.
6. `w_i′∝w_i exp(−αy_i h_i)`. ถูกเมื่อyh=1จึงคูณe^(−α),ผิดเมื่อyh=−1คูณe^α. ที่ε.2,αlog2จึงcorrect×.5,incorrect×2แล้วnormalizeให้sum1.
7. ถ้าL=.5(y−F)²จะdL/dF=F−yจึงnegativegradient=y−F. ถ้าใช้L=(y−F)²ได้2(y−F); constantfactorต้องถูกดูดเข้าηหรือfitstepให้ตรงนิยามก่อนเทียบimplementation.
8. ηเล็กมักต้องroundsมากขึ้นเพื่อให้effectivecapacityพอ แต่η×roundsไม่ใช่สมการequivalenceเพราะผู้เรียนแต่ละรอบfitresidualที่เปลี่ยน. Sweepร่วมvalidationและreportbestiterationกับbudget.
9. Histogramquantizefeatureเป็นbinsแล้วสะสมgradient/hessianต่อbin ลดcandidatecountและrepeatedsortcost. Binningเป็นapproximationที่ต้องfitจากtrain; จำนวนbinsแลกresolutionกับtime/memory.
10. Leaf-wiseเลือกleafที่ลดobjectiveมากที่สุดอาจสร้างกิ่งลึกและfitnoiseในข้อมูลน้อย. คุมnum_leaves,max_depth,min_data_in_leafและvalidationearlystop; ไม่สรุปจากtraininglossต่ำเพียงตัวเดียว.
11. Yieldpredictionหลังแต่ละstageโดยcopyหรือสร้างarrayใหม่ เพื่อไม่ให้ทุกhistoryentryอ้างbufferสุดท้าย. Teststaged[-1]==predictและstageก่อนหน้าไม่เปลี่ยนหลังfitroundถัดไป; เริ่มbaselineเป็นstage0ตามนิยามที่ประกาศ.
12. คำนวณvalidationlossหลังแต่ละround เก็บbestscore/bestiterationและหยุดเมื่อไม่ดีขึ้นpatienceรอบ แล้วrestorebestensemble. ห้ามใช้testearlystop; testcurveที่ดีถึงround3แล้วแย่ต้องคืนround3.
13. ใช้RNGเลือกsubsetrowsแบบwithoutreplacementต่อroundตามsubsamplefraction ขณะที่residualทุกsampleมาจากcurrentmodelเดิม. Testfraction1เท่ากับfullbatchและseedเดิมให้subsetsequenceเดิม; reportว่าตัวอย่างแต่ละรอบต่างจากbootstrap.
14. เก็บtrain/validationlossด้วยobjective/reductionเดียวกันทุกstageและplotจำนวนupdates. หากvalเริ่มขึ้นแต่trainลง ให้แสดงgapและbestiterationจริง ไม่smoothcurveจนซ่อนfailure.
15. คงsplit,η,roundsและlossแล้วเทียบdepth1กับdepth2; interactiondataเช่นXORช่วยแสดงcapacityต่างกัน. รายงานทั้งqualityและleafcount/runtime ไม่อ้างdepth2ชนะทุกdataset.
16. ถ้าε>.5สามารถflipbinarypredictionเมื่อสัญญารองรับแล้วใช้1−ε หรือreject/stoplearner. ε=.5ให้α0ไม่เพิ่มประโยชน์; testsตรวจweightupdatefiniteและnormalize ไม่ใช้negativevoteโดยไม่อธิบาย.
17. เลือกbestiterationจากvalidationก่อนจุดแย่ตามpatience ไม่ฝึกเพิ่มเพราะtrainlossยังลด. ตรวจleakage/noiseและcapacity; reporttestครั้งเดียวหลังfreezeprotocol ไม่เลือกiterationจากtestcurve.
18. CatBoostรองรับcategoricalfeaturesผ่านcat_featuresและorderedstatisticsตามAPI การone-hotทั้งหมดอาจสูญเสียประโยชน์/เพิ่มdimensionโดยไม่จำเป็น. เปรียบเทียบpipelineสองแบบบนsplitเดียวและตรวจtraining-onlycategoryprocessing ไม่ถือว่าone-hotผิดทุกกรณี.
19. ใช้logitsF,p=σ(F),negativegradient=y−p แล้วfitweaklearnerและเพิ่มηhในlogitspace. Stablelossคือmean(logaddexp(0,F)−yF); testzero-logitBCElog2และsmallstepลดlossบนข้อมูลtinyที่ตรวจได้.
20. Leafobjective≈G w+.5(H+λ)w²จึงstationary `w*=−G/(H+λ)`เมื่อH+λ>0. Gainต้องรวมregularizationและsplitpenaltyตามนิยาม; G=−4,H=2,λ2ให้w1และobjective−2ก่อนค่าคงที่.
