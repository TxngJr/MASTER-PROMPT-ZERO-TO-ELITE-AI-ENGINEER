/* เนื้อหาภาพจำลอง: แต่ละบทเลือกกลไกหลักหนึ่งเรื่อง แล้วเชื่อมไปยังบทเต็ม */
(function () {
  const lessons = [];
  function add(id, title, engine, parameter, steps, formula, misconception, quiz, experiment) {
    const [label, min, max, step, value] = parameter.split('|');
    const [question, answer, distractor, explanation] = quiz.split('|');
    lessons.push({ id, title, engine, parameter: { label, min: +min, max: +max, step: +step, value: +value },
      steps: steps.split('|').map(s => { const [title, detail] = s.split('~'); return {title, detail}; }),
      formula, misconception, quiz: {question, answer, distractor, explanation}, experiment });
  }
  add(1, 'โปรแกรมคิดตามคำสั่งอย่างไร', 'program', 'เกณฑ์กรองตัวเลข|1|9|1|4',
    'รับข้อมูล~รายการ [2, 4, 6, 8] เป็น input; ตัวแปรอ้างถึงข้อมูลที่โปรแกรมจะใช้|วนทีละตัว~loop อ่านสมาชิกตามลำดับ โดยยังไม่รวมตัวที่ไม่ผ่านเงื่อนไข|ตรวจเงื่อนไข~if x > threshold เลือก branch; เปลี่ยนเกณฑ์แล้วเส้นทางของข้อมูลเปลี่ยน|สะสมผลลัพธ์~ฟังก์ชันคืนผลรวมของตัวที่ผ่านเงื่อนไข; ตรวจผลด้วยตัวอย่างที่คำนวณมือได้',
    'total = sum(x for x in [2,4,6,8] if x > threshold)', 'เงื่อนไข > ไม่รวมค่าที่เท่ากับเกณฑ์ และ return ทำให้ฟังก์ชันจบ',
    'เมื่อ threshold = 4 ผลรวมเท่าไร|14 จาก 6 + 8|18 จาก 4 + 6 + 8|เพราะใช้ > จึงตัด 4 ออก; การอ่านเงื่อนไขให้ตรงช่วย debug branch ได้', 'ลองเกณฑ์ 4 และ 6 แล้วติดตามว่าตัวไหนถูกเลือกก่อนดูผลรวม');
  add(2, 'จากเวกเตอร์สู่อนุพันธ์', 'math', 'ตำแหน่ง x|-3|3|0.1|1',
    'แทนข้อมูล~เวกเตอร์ x และน้ำหนัก w มีมิติที่สอดคล้องกัน; dot product รวมผลคูณรายพิกัด|สร้างฟังก์ชัน~ใช้กรณีง่าย f(x)=x² เพื่อเห็นความสัมพันธ์ของ input และ output|วัดความชัน~อนุพันธ์ 2x บอกอัตราการเปลี่ยนเฉพาะบริเวณนั้น ไม่ใช่ค่าฟังก์ชัน|เชื่อมการเรียนรู้~gradient ชี้ทิศทางที่ loss เพิ่มเร็ว; optimizer จึงขยับสวนทางกับ gradient',
    'f(x) = x²; df/dx = 2x; x_next = x − η · 2x', 'gradient เป็นความชันเฉพาะจุด การก้าวใหญ่เกินไปอาจข้ามจุดต่ำสุด',
    'ที่ x = −2 gradient ชี้อย่างไร|มีค่า −4 จึงลด loss ด้วยการเพิ่ม x|มีค่า +4 จึงลด x|อนุพันธ์ของ x² คือ 2x; ลบ gradient ที่ติดลบทำให้ x เพิ่มเข้าใกล้ศูนย์', 'เลื่อน x ผ่านศูนย์แล้วดูเครื่องหมายของความชันเปลี่ยน');
  add(3, 'มอง shape แล้วเข้าใจ array', 'array', 'จำนวนแถวของ batch|1|6|1|3',
    'สร้าง array~batch มี shape (B,3); แต่ละแถวเป็นตัวอย่าง แต่ละคอลัมน์เป็น feature|ตรวจ broadcasting~bias shape (3,) ขยายไปตาม B แถวได้ เพราะมิติท้ายเท่ากัน|คำนวณพร้อมกัน~X + bias ทำงานรายพิกัดโดยไม่ต้องเขียน Python loop ทีละสมาชิก|ลดตามแกน~mean(axis=0) รวมตามแถวแล้วคืน 3 ค่า; axis ที่เลือกกำหนดความหมายของผล',
    'X: (B,3); bias: (3,); Y = X + bias; mean(Y, axis=0): (3,)', 'shape (B,1) กับ (B,) อาจ broadcast กลายเป็น (B,B) โดยไม่แจ้ง error',
    'mean(axis=0) ของ array (5,3) มี shape อะไร|(3,) เพราะลดแกนแถว|(5,) เพราะลดแกนคอลัมน์|axis=0 คือมิติแรก; การลดมิตินั้นเหลือค่าเฉลี่ยของแต่ละ feature', 'เปลี่ยน batch แล้วดู bias ถูกใช้ซ้ำและผลเฉลี่ยยังมี 3 ช่อง');
  add(4, 'ข้อมูลเดินผ่าน pipeline ที่ไม่รั่ว', 'data', 'สัดส่วน train (%)|50|90|5|70',
    'ตรวจข้อมูล~ตรวจ schema, missing values และ duplicate ก่อนนำตัวอย่างไปใช้|แบ่งชุดข้อมูล~แยก train และ test ตามหน่วยที่เป็นอิสระ; เลือกเวลา/กลุ่มเมื่อโจทย์ต้องการ|เรียน preprocessing~fit scaler และ imputer จาก train เท่านั้น เพื่อไม่อ่านสถิติของ test|แปลงและประเมิน~transform ทั้งสองชุดด้วยสถิติ train เดียวกัน แล้วประเมินบน test ที่กันไว้',
    'z_test = (x_test − μ_train) / σ_train', 'fit scaler ก่อน split ทำให้ test ส่งข้อมูลกลับไปสู่การฝึก เป็น data leakage',
    'ควร fit scaler บนชุดใด|เฉพาะ train|train รวมกับ test|test ต้องจำลองข้อมูลที่ยังไม่เห็น; ใช้ค่าเฉลี่ยจาก train เท่านั้น', 'เพิ่มสัดส่วน train แล้วดูจำนวนตัวอย่างที่เหลือสำหรับประเมิน');
  add(5, 'การเรียนรู้คือการลดความผิดพลาด', 'train', 'learning rate η|0.01|0.8|0.01|0.15',
    'ตั้งสมมติฐาน~เริ่มโมเดล ŷ=wx จากน้ำหนักเริ่มต้น; ยังไม่ได้เรียนจาก target|คำนวณ loss~MSE เปรียบเทียบ prediction กับ y; loss เป็นตัวเลขที่ใช้ฝึก|หา gradient~อนุพันธ์บอกว่า w มีผลต่อ loss อย่างไรบน training examples|อัปเดตและตรวจ~ปรับ w แล้วดู loss บนข้อมูลที่กันไว้; train loss ต่ำอย่างเดียวไม่ยืนยัน generalization',
    'L = mean((wx − y)²); ∂L/∂w = 2 · mean(x(wx−y)); w ← w−η∂L/∂w', 'test set ไม่ควรใช้เลือก hyperparameter ซ้ำ ๆ; ใช้ validation ก่อน test ครั้งสุดท้าย',
    'train loss ต่ำมาก แต่ validation สูง บ่งชี้อะไร|อาจ overfit และต้องตรวจ protocol|โมเดลดีเสมอเพราะจำ train ได้|โมเดลอาจจำ noise; validation วัดการทำงานกับข้อมูลนอกชุดฝึก', 'ลอง η เล็กและใหญ่ ดูว่าขั้นอัปเดตเข้าใกล้หรือกระโดดข้ามคำตอบ');
  add(6, 'เส้น regression ขยับเพราะอะไร', 'train', 'learning rate η|0.01|0.8|0.01|0.1',
    'วาด prediction~ในภาพใช้ ŷ=wx และ b=0 เพื่อแยกผลของ slope ให้ชัด|วัด residual~ระยะ y−ŷ ของแต่ละจุดแสดงข้อผิดพลาด; MSE ยกกำลังสองก่อนเฉลี่ย|รวม gradient~ตัวอย่างที่ x ใหญ่มีผลต่อ slope gradient มาก จึงต้องใส่ใจ feature scaling|ปรับเส้น~gradient descent เปลี่ยน w; บทเต็มเพิ่ม bias, polynomial และ regularization',
    '∂MSE/∂w = (2/n) Σ xᵢ(wxᵢ − yᵢ)', 'R² สูงไม่ได้พิสูจน์เหตุและผล และเส้นตรงอาจอธิบายความสัมพันธ์ที่ไม่เป็นเส้นตรงได้ไม่ดี',
    'เมื่อ prediction ต่ำกว่า y และ x > 0 gradient w เป็นอย่างไร|ติดลบ จึงเพิ่ม w เมื่อลบ gradient|บวก จึงลด w|x(wx−y) ติดลบ; การเพิ่ม slope ทำให้ prediction สูงเข้าใกล้ target', 'คาดเดาทิศที่เส้นจะขยับก่อนกดขั้นถัดไป');
  add(7, 'จาก logit สู่ความน่าจะเป็น', 'sigmoid', 'decision threshold|0.1|0.9|0.05|0.5',
    'คำนวณ logit~z=w·x+b เป็น score ที่ยังไม่จำกัดช่วง ไม่ใช่ probability|ผ่าน sigmoid~p=1/(1+exp(−z)) เปลี่ยน score เป็นค่าในช่วง 0 ถึง 1|ตัดสินด้วย threshold~p ≥ τ ให้ class 1; เปลี่ยน τ เปลี่ยน decision แต่ไม่เปลี่ยน p|ประเมินข้อผิดพลาด~threshold ต่ำมักเพิ่ม recall และ false positives; เลือกตามต้นทุนของโจทย์',
    'p = σ(z); class = 1[p ≥ τ]; BCE = −y log p − (1−y) log(1−p)', 'เปลี่ยน threshold ไม่ใช่การ retrain และ probability ของโมเดลไม่จำเป็นต้อง calibrated',
    'ลด threshold โดยคง prediction เดิม จะเกิดอะไร|จำนวน prediction class 1 ไม่ลดลง|probability ทุกจุดเพิ่มขึ้น|เราขยับกติกาตัดสินเท่านั้น ไม่ได้เปลี่ยน logits หรือ sigmoid', 'ลด τ จาก 0.7 เป็น 0.3 แล้วนับจุดที่เปลี่ยน class');
  add(8, 'KNN ให้เพื่อนบ้านโหวต', 'geometry', 'จำนวนเพื่อนบ้าน k|1|9|2|3',
    'เก็บตัวอย่าง~KNN เก็บ training points และ labels; ไม่มี weight vector ที่ต้อง fit แบบ linear model|คำนวณระยะ~วัดระยะ query ถึงทุกจุดหลัง scaling ที่เหมาะสม|เลือก k จุด~เรียงระยะแล้วเลือกเพื่อนบ้านที่ใกล้ที่สุด k ตัว|โหวต~class ที่มีเสียงมากสุดเป็น prediction; k เล็กไวต่อ noise ส่วน k ใหญ่ทำให้ขอบเขตเรียบ',
    'd(x,q) = √Σⱼ(xⱼ−qⱼ)²; ŷ = majority(y of k nearest)', 'หน่วย feature ที่ใหญ่สามารถครอบงำระยะ และการเพิ่ม k ไม่รับประกัน accuracy สูงขึ้น',
    'ทำไมต้อง scale feature ก่อน KNN|เพราะระยะขึ้นกับขนาดและหน่วยของ feature|เพราะ KNN ต้องใช้ gradient|KNN ใช้ระยะโดยตรง; feature หน่วยใหญ่ทำให้ลำดับเพื่อนบ้านเปลี่ยน', 'เปลี่ยน k แล้วดูวงเลือกเพื่อนบ้านและผลโหวต');
  add(9, 'Bayes อัปเดตความเชื่อจากหลักฐาน', 'bayes', 'prior P(class 1)|0.1|0.9|0.05|0.4',
    'กำหนด prior~P(class) เป็นความเชื่อก่อนเห็นหลักฐานใหม่|วัด likelihood~P(evidence given class) ถามว่าหลักฐานนี้พบได้มากแค่ไหนในแต่ละ class|คูณและ normalize~prior × likelihood ให้คะแนนแต่ละ class แล้วหารผลรวมเพื่อได้ posterior|จำแนก~เลือก posterior สูงสุด; Naive Bayes คูณ likelihood ของ features ภายใต้ conditional independence',
    'P(c given e) = P(e given c)P(c) / Σₖ P(e given k)P(k)', 'P(e given c) ไม่เท่ากับ P(c given e); อย่าสลับทิศของเงื่อนไข',
    'posterior ใช้ข้อมูลอะไร|ทั้ง prior และ likelihood|เฉพาะ likelihood โดยไม่สน prior|ฐานความถี่ของ class มีผลต่อคำตอบ แม้หลักฐานจะเหมือนเดิม', 'ลด prior ของ class 1 แล้วดูหลักฐานเดิมยังชนะได้หรือไม่');
  add(10, 'ต้นไม้ถามคำถามเพื่อลดความปน', 'tree', 'threshold ของการแบ่ง|1|9|1|5',
    'วัด parent~นับ class ใน node และคำนวณ Gini impurity; class เดียวมีค่า 0|ลอง split~เงื่อนไข x ≤ threshold แบ่งตัวอย่างเป็นสองกลุ่ม|เปรียบเทียบ gain~คำนวณ impurity ของลูกแบบถ่วงน้ำหนักด้วยจำนวนตัวอย่าง|เดินไป leaf~prediction ของ query เดินตาม branch; หยุดด้วย depth หรือขนาด leaf เพื่อควบคุม overfit',
    'Gini = 1−Σ p_c²; gain = G_parent−(n_L G_L+n_R G_R)/n', 'split ที่ลูกข้างหนึ่ง pure อาจไม่ใช่ split ที่ดีที่สุด ต้องดูค่าเฉลี่ยถ่วงน้ำหนักทั้งสองข้าง',
    'เหตุใดต้องถ่วง impurity ของลูกด้วยจำนวนจุด|เพื่อให้กลุ่มใหญ่มีน้ำหนักตามข้อมูลจริง|เพราะต้องให้ลูกสองข้างมีคะแนนเท่ากัน|node ที่มีข้อมูลมากมีผลต่อความผิดพลาดรวมมากกว่า node เล็ก', 'เลื่อน threshold แล้วหา gain สูงสุดจากข้อมูลในภาพ');
  add(11, 'หลายต้นไม้รวมเสียงลด variance', 'forest', 'จำนวนต้นไม้|1|9|1|5',
    'สุ่มแถว~bootstrap สุ่มตัวอย่างแบบคืนกลับให้แต่ละ tree เห็นข้อมูลต่างกัน|สุ่ม features~แต่ละ split ลอง subset ของ features ช่วยลดความสัมพันธ์ระหว่าง trees|ทำนายอิสระ~ภาพใช้คะแนนจาก trees จำลองเพื่อแสดงการรวมผล ไม่ได้ฝึก forest จริง|รวมเสียง~classification ใช้โหวตหรือเฉลี่ย probability; regression เฉลี่ยค่าทำนาย',
    'p̄ = (1/B) Σ_b p_b; correlated trees ลด variance ได้น้อยกว่า independent trees', 'เพิ่ม tree ไม่ได้ลบ bias และ OOB ไม่แทน time-based validation เมื่อข้อมูลมีลำดับเวลา',
    'random feature subsets ช่วยอะไร|ลดความคล้ายกันระหว่าง trees|ทำให้ทุก tree เหมือนกัน|trees ที่ผิดต่างกันมีโอกาสชดเชยกันเมื่อรวมผล', 'เพิ่ม B แล้วดูคะแนนเฉลี่ยและการสั่นของสมาชิกแต่ละตัว');
  add(12, 'Boosting แก้สิ่งที่โมเดลก่อนพลาด', 'boost', 'learning rate η|0.1|1|0.1|0.4',
    'เริ่ม baseline~โมเดลแรกทำนายค่าเริ่มต้นก่อนเพิ่ม weak learner|หา residual~กรณี squared loss ใช้ y−F เป็นทิศทางที่ learner ใหม่ต้องแก้|fit ตัวแก้~ภาพใช้ weak learner ที่แก้ได้ครึ่ง residual เพื่อเห็นการเรียนแบบต่อเนื่อง|รวมแบบย่อก้าว~F_new=F+ηh; η ลดขนาดการแก้ในแต่ละรอบ และต้องตรวจ validation',
    'F_m(x) = F_(m−1)(x)+η h_m(x); h_m ≈ negative loss gradient', 'Boosting ไม่ใช่การฝึก trees อิสระแล้วเฉลี่ย; residual ของ classification ขึ้นกับ loss',
    'learner รอบใหม่เรียนจากอะไรใน squared-loss boosting|ข้อผิดพลาดของ ensemble ปัจจุบัน|คำตอบของ tree แรกอย่างเดียว|ต้องคำนวณ residual หลังรวม learners ที่มีอยู่ทั้งหมด', 'ลด η แล้วดูการแก้ residual ช้าลงแต่ละรอบ');
  add(13, 'SVM มองระยะห่างจากเส้นแบ่ง', 'geometry', 'norm ของน้ำหนัก w|0.5|3|0.1|1',
    'สร้าง hyperplane~เส้น w·x+b=0 แบ่งสอง class และกำหนด signed score|หา support vectors~จุดใกล้ขอบ margin หรือผิดฝั่งมีบทบาทใน soft-margin objective|คำนวณ hinge~max(0,1−yz) ลงโทษจุดที่ยังไม่ผ่าน margin|ถ่วงสองเป้าหมาย~ลด norm(w) เพื่อเพิ่ม geometric margin พร้อมยอมจ่าย penalty สำหรับจุดผิด',
    'margin width = 2/norm(w); L = ½norm(w)² + CΣ max(0,1−yᵢzᵢ)', 'accuracy เดียวกันไม่ได้แปลว่า margin เท่ากัน; feature scaling เปลี่ยน geometry ของ SVM',
    'ถ้า norm(w) เพิ่ม โดยคงนิยาม canonical margin ความกว้างเป็นอย่างไร|แคบลงตาม 2/norm(w)|กว้างขึ้นตาม norm(w)|เส้นขอบ score ±1 อยู่ใกล้ hyperplane มากขึ้นเมื่อ norm ของ w ใหญ่', 'เพิ่ม norm แล้วดูเส้นขอบ margin เข้าใกล้เส้นแบ่ง');
  add(14, 'K-Means สลับจัดกลุ่มกับย้ายศูนย์', 'geometry', 'รอบ Lloyd iteration|0|6|1|1',
    'ตั้ง centroids~เริ่มศูนย์สองจุด; จุดเริ่มต้นต่างกันอาจพาไป local optimum ต่างกัน|assign~ให้ทุกจุดอยู่กลุ่มของ centroid ที่มี squared distance ต่ำสุด|update~ย้าย centroid ไปค่าเฉลี่ยของจุดในกลุ่มนั้น|ทำซ้ำ~สลับ assign และ update จนเสถียร; กลุ่มที่ได้ไม่ใช่ ground-truth class โดยอัตโนมัติ',
    'J = Σᵢ ||xᵢ−μ_(cᵢ)||²; μ_k = mean(x assigned to k)', 'K-Means ชอบกลุ่มแบบก้อนและไวต่อ scale; ไม่เหมาะกับรูปวงแหวนหรือ density ต่างกันทุกกรณี',
    'ขั้น update ย้าย centroid ไปที่ไหน|ค่าเฉลี่ยของสมาชิกในกลุ่ม|จุดที่ไกลที่สุดในกลุ่ม|ค่าเฉลี่ยลดผลรวม squared distances ของสมาชิกกลุ่มที่กำหนดไว้', 'เลื่อนรอบจาก 0 ไป 6 แล้วดู centroid และสมาชิกเปลี่ยนจนหยุด');
  add(15, 'PCA หมุนแกนเพื่อเก็บ variance', 'geometry', 'มุมแกน projection (องศา)|0|180|5|45',
    'center~ลบค่าเฉลี่ยจากแต่ละ feature ให้จุดหมุนอยู่ที่ origin|เลือกทิศ~unit vector u กำหนดแกนที่จะฉายข้อมูล; PCA เลือกทิศ variance สูงสุด|project~z=X_centered u ย่อข้อมูลสองมิติเป็นตำแหน่งบนเส้นหนึ่งมิติ|reconstruct~ฉายกลับ z uᵀ แล้วดูส่วนที่หายไป; explained variance บอกสัดส่วนที่เก็บได้',
    'z = X_centered u; variance(z) = uᵀ Cov(X)u; ||u||=1', 'PCA เก็บ variance มาก ไม่รับประกันเก็บข้อมูลที่สำคัญต่อ target มากที่สุด',
    'ทำไม PCA ต้อง center ก่อน|เพื่อวัดการกระจายรอบค่าเฉลี่ย|เพื่อบังคับทุกจุดเป็นศูนย์|เราเลื่อนข้อมูลทั้งกลุ่ม; ระยะสัมพันธ์ยังอยู่ แต่แกนวัด variance มี origin เหมาะสม', 'หมุนแกนแล้วหามุมที่ projection กระจายกว้างที่สุด');
  add(16, 'anomaly score ยังต้องมีเกณฑ์ตัดสิน', 'geometry', 'เกณฑ์ absolute z-score|1|3|0.1|2',
    'เรียนสถิติปกติ~กำหนดค่าเฉลี่ยและส่วนเบี่ยงเบนจากข้อมูลอ้างอิงที่เหมาะสม|คำนวณ score~abs(z) วัดระยะจาก mean ในหน่วย standard deviation|ตั้ง threshold~score เกิน threshold ถูก flag; ไม่ได้พิสูจน์ว่าเป็นความผิดปกติจริง|ตรวจ alert~ใช้ label หรือผู้เชี่ยวชาญประเมิน precision และ recall; ตรวจ distribution shift ด้วย',
    'z = (x−μ)/σ; alert = 1[abs(z) > τ]', 'outlier ไม่เท่ากับข้อมูลผิดเสมอ และ mean/std ไวต่อ outliers เอง',
    'เพิ่ม threshold จะมี alert อย่างไร|จำนวน alert ไม่เพิ่มขึ้น|ทุกจุดถูก flag เพิ่ม|เกณฑ์เข้มขึ้นทำให้จุดน้อยลงผ่านเงื่อนไข แต่บาง anomaly อาจหลุด', 'เลื่อน τ แล้วดูจุดปลายหางที่ถูกแจ้งเตือน');
  add(17, 'พยากรณ์ต้องเคารพลำดับเวลา', 'sequence', 'forecast horizon|1|8|1|3',
    'เรียงตามเวลา~อดีตมาก่อนอนาคต; random split อาจทำให้อนาคตรั่วมาสู่การฝึก|สร้าง lag~ใช้ y_(t−1) และอดีตที่รู้ได้จริงเมื่อถึงเวลาทำนาย|rollout~ภาพใช้ AR(1) coefficient 0.8; การพยากรณ์หลายก้าวใช้ผลทำนายก่อนหน้า|walk-forward~เลื่อนจุดตัดเวลาและประเมินหลายช่วง พร้อมเปรียบเทียบ naive baseline',
    'ŷ_(t+h) = 0.8ʰ y_t (AR(1) ตัวอย่างที่ไม่มี intercept)', 'ค่า actual ของอนาคตห้ามใช้เป็น lag ใน recursive forecast ที่ production ยังไม่เห็น',
    'recursive forecast ก้าวที่สองใช้อะไรแทน lag อนาคต|prediction ของก้าวแรก|actual ของอนาคตที่ยังไม่มาถึง|การประเมินต้องใช้ข้อมูลที่มี ณ เวลาพยากรณ์จริงเท่านั้น', 'เพิ่ม horizon แล้วดูค่า rollout และข้อจำกัดการทำนายหลายก้าว');
  add(18, 'neuron แปลงข้อมูลทีละชั้น', 'network', 'น้ำหนัก w|-2|2|0.1|0.8',
    'รับ input~ในภาพใช้ scalar x=2 เพื่ออ่านทุกตัวเลขได้; batch จริงใช้ matrix|linear~คำนวณ z=wx+b โดยกำหนด b=0.5|activation~ReLU(z)=max(0,z) เพิ่ม nonlinearity; z ติดลบจะถูกตัดเป็นศูนย์|output~คูณ hidden activation ด้วย v=1.5 เป็น prediction แล้วเปรียบเทียบ target y=3',
    'z=2w+0.5; h=max(0,z); ŷ=1.5h; L=(ŷ−3)²', 'ซ้อน linear layers โดยไม่มี nonlinearity ยังเทียบเท่า linear transformation เดียว',
    'ReLU ทำอะไรเมื่อ z < 0|คืน 0|คืน z เหมือนเดิม|ReLU ตัดส่วนลบ ทำให้ฟังก์ชันรวมของเครือข่ายไม่เป็นเส้นตรง', 'เลื่อน w จน z ติดลบแล้วดู hidden unit ปิด');
  add(19, 'backprop ส่งความรับผิดชอบย้อนกลับ', 'network', 'น้ำหนัก w|-0.2|2|0.1|0.8',
    'forward และเก็บค่า~คำนวณ z, h, prediction และ loss; backward ต้องใช้ค่าจาก forward เดียวกัน|seed gradient~เริ่ม dL/dL=1 และหา dL/dŷ=2(ŷ−y)|chain rule~ส่ง gradient ผ่าน output weight และ ReLU; ReLU ปิดให้ local gradient 0|ถึง parameter~คูณ dz/dw=x เพื่อได้ dL/dw; gradient ยังไม่เปลี่ยน w จน optimizer อัปเดต',
    'dL/dw = 2(ŷ−3) × 1.5 × 1[z>0] × 2', 'เมื่อ parameter ถูกใช้หลายทาง ต้องบวก gradients จากทุกเส้นทาง ไม่ใช่เขียนทับ',
    'backward เปลี่ยน weight ทันทีหรือไม่|ยังไม่เปลี่ยน ต้องให้ optimizer อัปเดต|เปลี่ยนทันทีเมื่อคำนวณ derivative|backprop คำนวณ sensitivity; optimizer ใช้ sensitivity ตัดสินขนาดก้าว', 'ปรับ w แล้วอ่าน gradient ย้อนจาก loss ไป input ทีละตัว');
  add(20, 'optimizer เปลี่ยน gradient เป็นก้าว', 'train', 'learning rate η|0.01|0.8|0.01|0.2',
    'รับ gradient~ใช้ objective ของ regression เล็ก ๆ เพื่อเห็นผลอัปเดตจริง|กำหนด learning rate~η ควบคุมระยะการขยับ ไม่ได้เปลี่ยน objective|SGD step~ลบ ηg ออกจาก parameter; ภาพแสดง SGD เพื่อแยกกลไกพื้นฐาน|ตรวจ trajectory~ดูหลายก้าวบน loss surface; Momentum และ Adam เพิ่ม state เพื่อเปลี่ยนวิธีเดิน',
    'SGD: θ_(t+1)=θ_t−ηg_t; Momentum: v_t=βv_(t−1)+g_t', 'AdamW weight decay ไม่เท่ากับใส่ L2 ใน gradient ของ Adam ทุกกรณี',
    'learning rate ใหญ่เกินไปอาจเกิดอะไร|loss สั่นหรือ diverge|ลด loss ได้เร็วขึ้นเสมอ|ก้าวที่ใหญ่เกินความโค้งของ loss surface กระโดดข้ามบริเวณที่เหมาะสม', 'ลอง η=0.1 กับ 0.8 แล้วเปรียบเทียบเส้นทาง loss');
  add(21, 'activation และ loss กำหนด gradient', 'activation', 'logit z|-6|6|0.2|1',
    'รับ logit~z เป็น score ก่อนแปลงเป็น probability|แปลง output~sigmoid ให้ p ในช่วง 0–1; ค่าใกล้ปลายช่วงมี derivative เล็ก|วัด BCE~target y=1 ให้ loss=−log p; ภาพใช้สูตร stable จาก logit|หา gradient~BCE คู่กับ sigmoid ให้ dL/dz=p−y; loss กับ activation ต้องเข้าคู่กับโจทย์',
    'p=σ(z); BCE(z,1)=max(z,0)−z+log(1+exp(−abs(z))); dL/dz=p−1', 'อย่าใช้ sigmoid ซ้ำก่อน BCEWithLogitsLoss และอย่าใช้ softmax ซ้ำก่อน CrossEntropyLoss',
    'ถ้า y=1 และ p ต่ำ gradient logit เป็นอย่างไร|ติดลบ เพื่อเพิ่ม logit|บวก เพื่อกด logit ลง|p−1 ติดลบ; การลบ gradient ทำให้ z เพิ่มและ p สูงขึ้น', 'เลื่อน z จาก −6 ไป 6 ดู p, loss และ gradient พร้อมกัน');
  add(22, 'PyTorch ทำ training loop อะไรบ้าง', 'network', 'น้ำหนัก w|-0.2|2|0.1|0.8',
    'เตรียม tensor~กำหนด shape, dtype, device และ requires_grad ให้ตรง|forward~model(x) สร้าง prediction และ graph; ภาพใช้ neuron scalar ที่อ่านมือได้|zero และ backward~ล้าง gradient ของรอบก่อน แล้ว loss.backward() สะสม gradient ของรอบนี้|optimizer step~optimizer.step() เปลี่ยน parameters; validation ใช้ eval และ no_grad ตามงาน',
    'zero_grad() → forward → loss → backward() → step()', 'eval() ไม่ได้ปิด autograd และ no_grad() ไม่ได้สลับ dropout เป็น eval',
    'ถ้าไม่ zero_grad ก่อนรอบใหม่จะเกิดอะไร|gradient สะสมข้ามรอบ|weight ถูกล้างเป็นศูนย์|PyTorch สะสม .grad โดยตั้งใจ; ต้องล้างเมื่อจบ accumulation ที่ต้องการ', 'เทียบตัวเลข forward กับ backward แล้วเชื่อมไป training loop ใน src');
  add(23, 'GradientTape จดกราฟให้ Keras', 'network', 'น้ำหนัก w|-0.2|2|0.1|0.8',
    'รับ batch~tf.data ป้อน inputs และ targets ที่ shape ถูกต้อง|บันทึก forward~GradientTape ติดตาม operations ที่เกี่ยวกับ trainable variables|หา derivatives~tape.gradient(loss, variables) ใช้ chain rule ย้อน graph|apply gradients~optimizer.apply_gradients อัปเดต weights; fit รวมขั้นเหล่านี้ไว้ให้',
    'with GradientTape: loss=f(variables); grads=tape.gradient(loss,variables)', 'การคำนวณผ่าน NumPy ระหว่าง graph อาจตัดเส้นทาง gradient',
    'GradientTape ต้องเห็นขั้นตอนไหน|forward ที่สร้าง loss|เฉพาะคำสั่ง print loss|derivative ต้องใช้ความสัมพันธ์ของ operations ที่พา variable ไปสู่ loss', 'เปลี่ยน w แล้วคำนวณ gradient มือเทียบกับที่ tape ควรให้');
  add(24, 'kernel เลื่อนอ่านภาพทีละช่อง', 'conv', 'ตำแหน่ง kernel|0|8|1|0',
    'รับภาพ~ภาพ 5×5 ในตัวอย่างมีขอบแนวตั้ง; kernel 3×3 ใช้น้ำหนักชุดเดิมทุกตำแหน่ง|เลือก patch~เลื่อนช่องอ่านโดย stride=1 และไม่มี padding|คูณแล้วรวม~framework ส่วนใหญ่ทำ cross-correlation: คูณ patch กับ kernel โดยไม่พลิก kernel|สร้าง feature map~ผลแต่ละตำแหน่งเรียงเป็น 3×3; activation และ pooling ใช้ต่อได้',
    'Y[i,j]=Σ_ab X[i+a,j+b]K[a,b]; H_out=(5−3)/1+1=3', 'kernel ที่ใช้ซ้ำคือ weight sharing; channels ไม่ใช่จำนวนตำแหน่งที่ kernel เลื่อน',
    'ภาพ 5×5 กับ kernel 3×3 stride 1 ไม่มี padding ได้ขนาดเท่าไร|3×3|5×5|(H−K)/S+1 ให้ 3; padding เท่านั้นที่อาจช่วยคงขนาด', 'เลื่อนตำแหน่งแล้วอ่านผลคูณของ patch กับ kernel');
  add(25, 'RNN ใช้ state เดิมอ่านคำถัดไป', 'sequence', 'น้ำหนัก recurrent|0|1.5|0.1|0.8',
    'เริ่ม state~h₀=0 สรุปอดีตที่ยังว่าง|รับ input ทีละเวลา~ลำดับ [1, −0.5, 0.8, 0.2] ใช้ weights ชุดเดียวกัน|อัปเดต hidden~h_t=tanh(x_t+w h_(t−1)); แต่ละ state ขึ้นกับอดีต|ส่ง output~hidden ล่าสุดสรุปลำดับ; BPTT ส่ง gradient ย้อนผ่านทุกเวลาที่ unroll',
    'h_t=tanh(x_t+w h_(t−1)); weights shared across t', 'state ที่ใช้ร่วมกันไม่ใช่ weight; gradient ผ่านหลายเวลาอาจหายหรือระเบิด',
    'RNN มี recurrent weight ใหม่ทุก timestep หรือไม่|ใช้ชุดเดียวกันทุกเวลา|มี weight ใหม่เสมอ|weight sharing ทำให้ใช้กลไกเดียวกันกับลำดับหลายความยาว', 'ลด w เป็น 0 แล้วดูว่าแต่ละ state ไม่รับอดีตแล้ว');
  add(26, 'gate คุมการเก็บและลืมความจำ', 'gates', 'forget gate f|0|1|0.05|0.8',
    'อ่าน state เก่า~cell state เริ่ม c₀=1 เป็นความจำที่ส่งต่อ|เปิด forget gate~f ในช่วง 0–1 คูณความจำเก่าเพื่อเลือกสัดส่วนที่เก็บ|เพิ่มข้อมูล~i·g เพิ่ม candidate ใหม่; ภาพแยกเส้นทางเก็บความจำโดยตั้ง candidate=0|ส่ง state ต่อ~เมื่อไม่เพิ่มข้อมูล c_t=fᵗc₀; บทเต็มรวม output gate และ GRU update/reset gates',
    'c_t=f_t c_(t−1)+i_t g_t; h_t=o_t tanh(c_t)', 'forget gate ไม่ได้ตัดสินแบบ 0 หรือ 1 เท่านั้น; ค่า 0.9 ก็ทำให้ข้อมูลลดลงเมื่อผ่านหลายเวลา',
    'เมื่อ f=1 และ i·g=0 ความจำเปลี่ยนอย่างไร|คงค่าเดิม|ถูกล้างเป็นศูนย์|เส้นทาง additive ของ cell state สามารถส่งความจำเดิมต่อโดยไม่ลดขนาด', 'เปรียบเทียบ f=0.5 และ 0.95 ใน 8 timestep');
  add(27, 'VAE สุ่มผ่าน latent ที่เรียนได้', 'latent', 'noise ε|-2|2|0.1|0.5',
    'encode~encoder ให้ μ=0.5 และ σ=0.8 ในตัวอย่าง แทน distribution ของ latent|reparameterize~สุ่ม ε จาก standard normal แล้วสร้าง z=μ+σε; slider เลือกตัวอย่าง ε|decode~decoder ตัวอย่างใช้ x̂=2z เพื่อให้เห็นผลของ latent ต่อ output|ฝึกสองเป้าหมาย~reconstruction ให้ข้อมูลกลับมา; KL ทำ posterior ใกล้ prior ที่ใช้สุ่มตอนสร้างข้อมูล',
    'z=μ+σε; ε~N(0,1); L=reconstruction+β KL(q(z given x) || N(0,1))', 'VAE ไม่ใช่ autoencoder ที่แค่ใส่ noise; KL และ distribution ของ latent เป็นส่วนของ objective',
    'ทำไมใช้ z=μ+σε|ให้ gradient ไหลผ่าน μ และ σ ได้|เพื่อหยุด gradient ทุกตัว|ความสุ่มย้ายไป ε; μ และ σ อยู่ในเส้นทาง differentiable', 'เลื่อน ε แล้วดู latent sample กับ reconstruction เปลี่ยน');
  add(28, 'GAN เรียนด้วยเกมสองฝ่าย', 'gan', 'logit ของ D(fake)|-4|4|0.2|0',
    'generate~G เปลี่ยน noise เป็น fake sample โดยยังต้องการ gradient|ฝึก D~D แยก real กับ fake; ตัด graph ของ G เมื่ออัปเดตเฉพาะ D|ฝึก G~ใช้ D ให้สัญญาณว่า fake ยังไม่น่าเชื่อ; ไม่ detach fake ในขั้นนี้|สลับรอบ~ภาพแสดง non-saturating loss ของ G; คุณภาพต้องตรวจ coverage และ mode collapse ด้วย',
    'L_G=−log σ(D_logit(G(z))); dL_G/dlogit=σ(logit)−1', 'detach fake ตอนฝึก G ตัด gradient ไป generator ทำให้ G เรียนไม่ได้',
    'ฝึก D ควร detach fake เพื่ออะไร|ไม่ให้ update graph ของ G ในรอบ D|เพื่อให้ G เปลี่ยนทันที|การสลับ optimization ต้องแยก parameter และ graph ของฝ่ายที่กำลังฝึก', 'เลื่อน D(fake) ดูความมั่นใจของ discriminator กับแรง gradient ที่ G ได้');
  add(29, 'Attention อ่านข้อมูลตามความเกี่ยวข้อง', 'attention', 'ตัวหาร score เพิ่มเติม|0.5|3|0.1|1',
    'สร้าง Q K V~query และ keys ใช้เปรียบเทียบ; values เป็นข้อมูลที่จะถูกรวม เป็น learned projections|dot product~คำนวณ q·k แล้วหาร √d_k; slider เพิ่ม divisor เพื่อสำรวจความคมของ weights|softmax~ลบ row max ก่อน exponentiate แล้ว normalize ให้ weights รวมเป็น 1|weighted sum~output=Σ a_j v_j; weight สูงทำให้ value ตำแหน่งนั้นมีส่วนมาก',
    'A=softmax(QKᵀ/√d_k); O=AV; Σ_j A_ij=1', 'attention weight ไม่ใช่หลักฐานอธิบายเหตุและผลของ prediction โดยตรง',
    'ผลรวม attention weights ต่อ query เท่าไร|1 สำหรับแถวที่มีตำแหน่งให้ attend|จำนวน tokens|softmax normalize ในแกน key; แถวที่ถูก mask ทั้งหมดต้องจัดการเป็นพิเศษ', 'เพิ่ม divisor แล้วดู weights กระจายสม่ำเสมอขึ้น');
  add(30, 'Transformer ส่งข้อมูลผ่าน residual stream', 'attention', 'ตัวหาร attention scores|0.5|3|0.1|1',
    'embed + position~token IDs เปลี่ยนเป็น vectors และรับข้อมูลตำแหน่ง|attention~แต่ละ token อ่าน values ของตำแหน่งที่อนุญาต; ภาพแยกหนึ่ง query เพื่ออ่านตัวเลข|residual + MLP~บวกผล attention กลับ stream แล้วผ่าน normalization และ nonlinear FFN|project logits~LM head ให้ score ต่อ vocabulary; decoder-only ใช้ causal mask และ next-token loss',
    'pre-norm: H=X+Attention(LN(X)); Y=H+FFN(LN(H))', 'Transformer ไม่ใช่ attention อย่างเดียว; residual, norm, FFN และ position มีบทบาทต่างกัน',
    'FFN แบบ position-wise แชร์ weights อย่างไร|ชุดเดียวกันทุก token|ใหม่หนึ่งชุดต่อ token|แต่ละตำแหน่งผ่านฟังก์ชันเดียวกัน แต่รับ representations ที่ต่างกัน', 'ดู attention sublayer แล้วตาม flow ไป residual และ FFN ในบทเต็ม');
  add(31, 'ViT เปลี่ยนภาพให้เป็น tokens', 'patch', 'ขนาด patch (pixels)|2|8|2|4',
    'แบ่งภาพ~ภาพ 16×16 ถูกแบ่งเป็น patches ที่ไม่ทับกัน|flatten + project~patch P×P×C เปลี่ยนเป็น vector แล้ว linear project ไป dimension D|ใส่ตำแหน่ง~เพิ่ม positional embedding และอาจใช้ CLS token เพื่อรวมข้อมูล|encoder + head~self-attention ให้ patches แลกข้อมูล; head ทำนาย class จาก CLS หรือ pooling',
    'N=(H/P)(W/P); patch vector size=P²C; dense attention pairs=N²', 'patch ใหญ่ลด token count แต่สูญเสียรายละเอียดบางส่วน; CLS ไม่ได้เป็นข้อบังคับทุกสถาปัตยกรรม',
    'ลด patch size ครึ่งหนึ่งในภาพสองมิติ tokens เพิ่มเท่าไร|4 เท่า|2 เท่า|ทั้งแนวสูงและกว้างเพิ่มจำนวน patches สองเท่า จึงคูณกันเป็น 4', 'เปลี่ยน P แล้วดู grid กับจำนวนคู่ attention โตเร็ว');
  add(32, 'ภาษาเป็น tokens ก่อนเป็นโมเดล', 'tokens', 'ความยาว context n-gram|1|4|1|2',
    'normalize~กำหนดนโยบาย Unicode และ whitespace ให้สม่ำเสมอโดยไม่ลบความหมายสำคัญ|tokenize~ภาพใช้ตัวอักษรของ banana เป็นตัวอย่าง ไม่ใช่ tokenizer ภาษาไทยสำเร็จรูป|สร้าง context-target~เลือก n tokens ก่อนหน้าเพื่อทำนาย token ถัดไป|นับและประเมิน~n-gram probability ต้อง smoothing เมื่อคู่ไม่เคยพบ; vocabulary fit จาก train',
    'P(next given context)=(count(context,next)+k)/(count(context)+kV)', 'ภาษาไทยไม่สามารถแบ่งคำได้ถูกต้องด้วย whitespace อย่างเดียว',
    'token IDs เป็น semantic vectors แล้วหรือไม่|ยังเป็นดัชนี ต้อง lookup embedding|เป็นเวกเตอร์ความหมายทันที|ID เป็นตัวระบุเชิงสัญลักษณ์; ระยะระหว่าง IDs ไม่มีความหมายเชิงภาษา', 'เพิ่ม context แล้วดูคู่ input-target เปลี่ยน');
  add(33, 'embedding เรียนจากบริบทของคำ', 'geometry', 'มุม query vector (องศา)|0|180|5|30',
    'สร้างคู่คำ~Skip-Gram ใช้ center word ทำนาย context; negatives เป็นคู่เปรียบเทียบ|lookup vectors~input และ output embedding tables เปลี่ยน IDs เป็น vectors ที่เรียนได้|ปรับความสัมพันธ์~objective เพิ่ม score คู่จริงและลดคู่ลบ; ภาพสำรวจ cosine หลังได้ vectors แล้ว|ใช้ similarity~cosine วัดมุม ไม่ใช่ความยาว; static embedding ให้คำเดียว vector เดียวแม้บริบทเปลี่ยน',
    'cos(u,v)=(u·v)/(||u||||v||); SGNS uses positive and negative pairs', 'cosine สูงไม่ได้ยืนยันว่าคำมีความหมายเหมือนกันทุกบริบท และ analogy ไม่ใช่กฎแน่นอน',
    'คูณ vector ด้วย 2 cosine เปลี่ยนหรือไม่|ไม่เปลี่ยนถ้าคูณด้วยค่าบวก|เพิ่มเป็นสองเท่า|cosine หารด้วย norms จึงวัดมุมหลัง normalization', 'หมุน query แล้วดูอันดับคำใกล้เคียงเปลี่ยน');
  add(34, 'BERT อ่านรอบด้านเพื่อเติมช่องว่าง', 'attention', 'ตำแหน่ง query token|1|4|1|2',
    'corrupt tokens~เลือกตำแหน่งสำหรับ MLM และเก็บ original token เป็น target|bidirectional encode~query อ่านซ้ายและขวาทุกตำแหน่งที่ไม่ใช่ padding|predict masked positions~LM head ทำนาย original IDs; loss ใช้เฉพาะตำแหน่งที่เลือก|fine-tune~representation ใช้กับ classification หรือ token tasks; MLM ไม่ใช่ autoregressive generation',
    'L_MLM=−Σ_(i selected) log P(x_i given corrupted sequence)', 'BERT ใช้ padding mask แต่ไม่ใช้ causal mask แบบ GPT; ไม่ใช่ทุก token ใน MLM ที่ถูกแทนด้วย MASK',
    'BERT อ่าน token ด้านขวาได้หรือไม่|ได้ใน bidirectional encoder|ไม่ได้เพราะทุก Transformer เป็น causal|รูปแบบ mask ขึ้นกับสถาปัตยกรรม ไม่ใช่คำว่า Transformer', 'เปลี่ยน query แล้วดูแถว attention อ่านได้ทั้งสองด้าน');
  add(35, 'GPT มองอดีตและสร้าง token ถัดไป', 'attention', 'ตำแหน่ง query token|1|4|1|2',
    'shift targets~input [A,B,C] คู่กับ target [B,C,D] ใน next-token training|causal attention~ตำแหน่ง t อ่านได้ถึง t เท่านั้น; ช่องอนาคตถูก mask ก่อน softmax|predict distribution~LM head ให้ logits แล้ว softmax/sampling เลือก token ถัดไป|append และวน~เติม token ที่สร้างลง context แล้วทำซ้ำจน EOS หรือ budget; KV cache ลดคำนวณส่วนเดิม',
    'P(x₁:T)=Π_t P(x_t given x_<t); causal mask forbids j>i', 'training parallel ทุกตำแหน่งได้เพราะมี causal mask; inference ยังสร้างทีละ token',
    'ทำไมต้อง mask อนาคตระหว่าง training|ไม่ให้เห็น target ที่ต้องทำนาย|เพื่อลด vocabulary|หากอ่านอนาคตได้ loss อาจต่ำจาก leakage แต่ใช้ตอนสร้างจริงไม่ได้', 'เปลี่ยน query แล้วดูสามเหลี่ยมตำแหน่งที่อนุญาต');
  add(36, 'T5 เชื่อม encoder กับ decoder', 'attention', 'ตำแหน่ง decoder query|1|4|1|2',
    'encode source~encoder อ่าน source แบบ bidirectional และส่ง representations|decoder self-attention~target ที่ shift right ใช้ causal mask อ่านเฉพาะอดีต target|cross-attention~decoder Q อ่าน encoder K,V; ความยาว source และ target อาจต่างกัน|generate target~LM head ทำนาย target ถัดไป แล้ววนแบบ autoregressive; T5 ใช้ text-to-text',
    'cross attention: Q=H_decoder W_Q; K,V=H_encoder W_K,V', 'cross-attention ไม่ใช้ causal mask ของ source โดยปกติ แต่ต้อง mask source padding',
    'cross-attention values มาจากไหน|encoder representations|target ในอนาคต|decoder อ่านข้อมูล source ที่ encode ไว้ โดยใช้ query จากตัวเอง', 'เปลี่ยน decoder query แล้วติดตามน้ำหนักการอ่าน source');
  add(37, 'GNN ส่ง message ตามขอบกราฟ', 'graph', 'รอบ message passing|0|4|1|1',
    'ตั้ง node features~แต่ละ node มีค่าเริ่มต้นและ neighbors จาก edges|ส่ง messages~neighbor ส่ง representation มาตาม edge ไม่ได้ส่งให้ทุก node อัตโนมัติ|aggregate~ภาพเฉลี่ย features ของตัวเองและ neighbors ให้ไม่ขึ้นกับลำดับ|update ซ้ำ~แต่ละรอบขยาย receptive field; หลายรอบเกินไปอาจ oversmooth',
    'h_i^(l+1)=mean({h_i^l} ∪ {h_j^l : j in N(i)}) [toy mean aggregation]', 'GNN ไม่ได้ใช้ attention เสมอ; aggregation ต้องไม่ขึ้นกับลำดับรายชื่อ neighbors',
    'สองรอบส่ง message อ่านข้อมูลได้ไกลเท่าไร|ถึง neighbors สอง hops|ทุก node เสมอ|หนึ่ง layer ส่งข้อมูลได้หนึ่ง edge; ต้องมีเส้นทางเชื่อมจึงเข้าถึงได้', 'เพิ่มจำนวนรอบแล้วดูค่า node ค่อย ๆ ใกล้กัน');
  add(38, 'RL เรียนจาก reward หลังลงมือ', 'reward', 'discount factor γ|0|1|0.05|0.9',
    'สังเกต state~agent รับ state ของ environment และเลือก action จาก policy|environment step~action ทำให้เกิด state ใหม่และ reward; ไม่มี label ของ action ที่ดีที่สุดเสมอ|คำนวณ return~รวม reward ในอนาคตแบบ discount; ภาพใช้ [0,1,3,5]|ปรับ policy/value~Q-learning ใช้ TD target; exploration ทำให้เก็บประสบการณ์ที่ยังไม่เคยลอง',
    'G=Σ_t γᵗr_t; Q←Q+α[r+γ max_a Q(s_next,a)−Q]', 'reward สูงระยะสั้นอาจไม่ให้ return ดีระยะยาว และ terminal state ต้องไม่ bootstrap ต่อ',
    'ถ้า γ=0 return สนใจอะไร|reward ตอนนี้เท่านั้น|reward อนาคตทั้งหมดเท่ากัน|terms หลังปัจจุบันถูกคูณด้วย 0; policy จึงไม่มีแรงจาก delayed rewards', 'เปลี่ยน γ แล้วดู contribution ของ reward แต่ละเวลา');
  add(39, 'generative model เรียน distribution', 'latent', 'latent sample z|-2|2|0.1|0.5',
    'เก็บ distribution~ข้อมูลตัวอย่างสะท้อนบางส่วนของ distribution ที่ต้องการเรียน|กำหนดกลไก~ภาพใช้ latent decoder x=2z เพื่อแยกการสุ่มออกจากการจำแนก|sample~สุ่ม latent จาก prior แล้วแปลงเป็นตัวอย่างใหม่|ประเมินหลายด้าน~ตรวจ fidelity, coverage, diversity และ memorization; likelihood อย่างเดียวไม่พอ',
    'z~p(z); x=G(z); autoregressive family: p(x)=Π_t p(x_t given x_<t)', 'generator ที่ภาพดูสวยอาจครอบคลุม modes ไม่ครบ; generative ไม่ได้แปลว่าไม่จำข้อมูลฝึก',
    'โมเดลสร้างได้แบบเดียวแต่ดูดี ขาดคุณสมบัติใด|coverage และ diversity|ความละเอียดของทุกภาพเท่านั้น|mode collapse ไม่ครอบคลุม distribution แม้บาง sample ดูสมจริง', 'เปลี่ยน z แล้วแยกความหมายระหว่าง sample กับ probability distribution');
  add(40, 'diffusion เติม noise แล้วเรียนย้อนกลับ', 'noise', 'ระดับ noise t|0|10|1|4',
    'ภาพเริ่มต้น~x₀ เป็น pattern 8×8 ที่สะอาด; noise ถูกกำหนดคงที่เพื่อเปรียบเทียบได้|forward noising~ผสม √ᾱ_t x₀ กับ √(1−ᾱ_t)ε; t เพิ่มสัดส่วน noise|เรียนทำนาย noise~โมเดลรับ x_t และ t เพื่อทำนาย ε; ภาพยังไม่รันโมเดลที่ฝึกจริง|reverse sampling~เริ่มจาก noise แล้วใช้ learned reverse steps ลด noise ทีละรอบ ไม่ใช่เปิดภาพต้นฉบับกลับเฉย ๆ',
    'x_t=√ᾱ_t x₀+√(1−ᾱ_t)ε; L=E||ε−ε_θ(x_t,t)||²', 'training ใช้ noisy sample หนึ่งเวลาได้ แต่ generation ต้องเดิน reverse process หลายขั้น',
    'โมเดล noise-prediction ต้องรับ timestep เพื่ออะไร|เพื่อรู้ระดับ noise ของ input|เพื่อเลือกชื่อไฟล์|distribution ของ x_t เปลี่ยนตาม t; การทำนายต้องรู้เงื่อนไขการเติม noise', 'เพิ่ม t แล้วดู signal-to-noise ratio ลดลง');
  add(41, 'ภาพและข้อความพบกันใน embedding space', 'similarity', 'temperature τ|0.1|2|0.1|0.5',
    'encode สองฝั่ง~image encoder และ text encoder เปลี่ยนคนละ modality เป็น vectors|normalize~หารด้วย norm เพื่อใช้ cosine similarity ใน space เดียวกัน|สร้าง similarity matrix~จับคู่ภาพกับข้อความทุกคู่ใน batch; คู่ตรงกันเป็น positives|contrastive loss~เพิ่ม probability ของคู่ตรงและลดคู่ผิด; ภาพใช้ scores จำลองที่กำหนดไว้',
    'S_ij=cos(image_i,text_j)/τ; L_i=−log softmax(S_i)[matching j]', 'encoder สองตัวต้องเรียน alignment; dimension เท่ากันไม่ได้ทำให้ space ตรงกันเอง',
    'ลด τ ทำให้ softmax เป็นอย่างไร|คมขึ้นเมื่อ scores ไม่เท่ากัน|สม่ำเสมอขึ้น|หารด้วยค่าที่เล็กเพิ่มช่องว่าง logits ทำให้ probability กระจุกที่ score สูง', 'เปลี่ยน τ แล้วดู probability ของคู่ภาพ-ข้อความ');
  add(42, 'เสียงเปลี่ยนเป็น frames ก่อนเข้าโมเดล', 'audio', 'ขนาด frame (samples)|8|32|8|16',
    'รับ waveform~amplitude ตามเวลา; sample rate กำหนดระยะเวลาต่อหนึ่ง sample|แบ่ง frame~window สั้นช่วยอ่านความถี่ที่เปลี่ยนตามเวลา|แปลง features~STFT แปลง frame ไป frequency bins แล้ว Mel/log-Mel เป็น features|decode หรือ synthesize~STT ให้ข้อความ ส่วน TTS ใช้ข้อความสร้าง acoustic features แล้ว vocoder คืน waveform',
    'frame duration=N/sample_rate; Δf=sample_rate/N', 'Mel spectrogram ไม่ใช่ข้อความ และ sample rate ต่างกันต้อง resample อย่างถูกต้อง',
    'frame ใหญ่ขึ้นทำให้ frequency spacing อย่างไร|ละเอียดขึ้น เพราะ Δf ลดลง|หยาบขึ้นเพราะ Δf เพิ่ม|N เพิ่มทำให้ sample_rate/N ลด แต่ time resolution ลดลงด้วย', 'ปรับ N แล้วดูกรอบเวลาของ frame และความละเอียดเชิงความถี่');
  add(43, 'recommendation เริ่มที่ retrieval แล้ว ranking', 'similarity', 'temperature สำหรับแสดงคะแนน|0.1|2|0.1|0.5',
    'สร้าง user-item data~observed interactions สะท้อนทั้ง preference และโอกาสได้เห็นสินค้า|embed สองฝั่ง~user และ item vectors เรียนจาก feedback; ภาพใช้ dot scores ที่กำหนดไว้|retrieve~เลือก candidates จาก similarity เพื่อคุมค่าใช้จ่าย|rank และตรวจ~ranker เพิ่ม context; ใช้ temporal split และ ranking metrics; softmax ในภาพเป็นตัวสอน ไม่ใช่ serving requirement',
    'score(u,i)=u·v_i; NDCG@K=DCG@K/IDCG@K', 'unobserved item ไม่ใช่ negative แท้เสมอ และ popularity bias อาจกีดกันสินค้าใหม่',
    'ทำไม retrieval มาก่อน ranking|ลด candidates ก่อนใช้ model ที่แพงกว่า|เพราะ ranking ไม่ใช้ features|retrieval หาผู้สมัครจำนวนจำกัด; ranker ประเมินรายละเอียดที่แพงกว่าได้', 'ดูอันดับ item และแยก raw score ออกจาก probability ที่แปลงไว้สอน');
  add(44, 'boxes ซ้อนกันแค่ไหนก่อน NMS', 'boxes', 'ระยะเลื่อน predicted box|0|100|5|25',
    'ทำนาย box~detector ให้ coordinates, class และ confidence|หาส่วนทับ~intersection เป็นพื้นที่ที่ prediction และ ground truth ซ้อนกัน|คำนวณ IoU~หาร intersection ด้วย union เพื่อวัดตำแหน่งที่ตรงกัน|NMS / evaluate~NMS คัดกล่องซ้ำตาม score และ IoU; segmentation ทำนายละเอียดถึง pixel',
    'IoU=intersection/(area_A+area_B−intersection)', 'confidence สูงไม่ได้แปลว่า box ตรง และ NMS threshold ไม่ใช่ confidence threshold',
    'กล่องตรงกันทุกพิกัดมี IoU เท่าไร|1|0|intersection เท่ากับ union เมื่อกล่องมีพื้นที่บวกและตรงกัน', 'เลื่อน box ดู intersection, union และ IoU เปลี่ยนจริง');
  add(45, 'vector search หาใกล้โดยไม่อ่านทุกเอกสาร', 'geometry', 'จำนวนผล top-k|1|6|1|3',
    'embed query~ใช้ embedding model รุ่นและ dimension ที่ตรงกับ index|วัด similarity~ภาพใช้ exact Euclidean distance เป็น baseline ที่ตรวจคำตอบได้|เลือก top-k~เรียง candidate ตามระยะ; ANN แลก recall กับเวลาและ memory|filter และ rerank~ตรวจ metadata/สิทธิ์ก่อนส่ง context; ประเมิน ANN เทียบ exact neighbors',
    'exact top-k=arg smallest distances; ANN recall@k=|ANN_k ∩ exact_k|/k', 'เปลี่ยน embedding model แล้วใช้ index เดิมอาจให้ space ที่ไม่ตรงกันแม้ dimension เท่ากัน',
    'ANN recall ควรเทียบกับอะไร|exact search ในข้อมูลชุดเดียวกัน|จำนวนเอกสารทั้งหมดอย่างเดียว|exact neighbors เป็น baseline เพื่อวัดว่าการค้นประมาณพลาดอะไร', 'เพิ่ม k แล้วดู candidate ที่เข้ามาไกลขึ้น');
  add(46, 'RAG ต้องค้นหลักฐานก่อนตอบ', 'retrieval', 'context budget (tokens)|20|100|10|60',
    'index เอกสาร~parse, chunk, embed พร้อม metadata; ในภาพเตรียม candidates ที่จัดอันดับแล้ว|retrieve~query หา chunk ที่เกี่ยวข้อง; ขั้นนี้แยกจาก generation|จัด context~เลือก chunks ตาม ranking ภายใต้ token budget และสิทธิ์การเข้าถึง|ตอบพร้อมหลักฐาน~generator อ่าน context และอ้างอิง chunk; หากไม่มีหลักฐานพอต้องบอกข้อจำกัด',
    'Σ tokens(selected chunks) ≤ context budget', 'RAG ไม่ได้แก้ hallucination โดยอัตโนมัติ; retrieval miss และ context ที่ผิดยังทำให้ตอบผิด',
    'เพิ่ม context budget รับประกันคำตอบดีขึ้นหรือไม่|ไม่เสมอ อาจเพิ่มข้อมูลรบกวน|ดีขึ้นเสมอทุก query|ต้องประเมิน retrieval relevance, context selection และ grounded answer แยกกัน', 'ปรับ budget แล้วดู chunks ที่เข้า context และที่ถูกตัดออก');
  add(47, 'agent วน model → tool → observation', 'agent', 'step budget|1|6|1|4',
    'รับงานและ state~เป้าหมาย, messages และ budget เป็น state ของวงจร|model เสนอ tool call~เลือกชื่อ tool และ arguments; model ไม่ใช่ executor|validate และ execute~executor ตรวจ schema/สิทธิ์แล้วรัน tool; ภาพจำลอง tool อ่านข้อมูล|observe และหยุด~ผล tool กลับเข้า state; หยุดเมื่อได้คำตอบหรือหมด budget',
    'state → model → validated tool → observation → state; steps ≤ budget', 'tool output เป็นข้อมูลที่ไม่เชื่อถือ ไม่ใช่คำสั่งระดับ system',
    'ใครรับผิดชอบรัน tool จริง|executor หลังตรวจ call|ข้อความของ model โดยตรง|tool call เป็นข้อเสนอแบบมีโครงสร้าง; application คุมขอบเขตการลงมือจริง', 'ลด budget แล้วดูวงจรหยุดก่อนถึงคำตอบที่จำลองไว้');
  add(48, 'decoder block กับขนาด KV cache', 'cache', 'จำนวน KV heads|1|8|1|8',
    'embed และ norm~token representations วิ่งใน residual stream; RMSNorm ปรับ scale|Q K V + RoPE~project heads และเพิ่มตำแหน่งผ่านการหมุน Q,K|attention + SwiGLU~รวม information แล้วผ่าน gated FFN; residual บวกกลับทุก sublayer|cache ตอน decode~เก็บ K,V ของอดีต; GQA/MQA ลด KV heads ได้โดยคง query heads',
    'KV bytes=2 × L × B × T × H_kv × D_head × bytes_per_element', 'KV cache เป็น activation state ของ request ไม่ใช่ model weights และโตตามความยาว context',
    'ลด KV heads จาก 8 เป็น 2 cache เปลี่ยนอย่างไร|เหลือหนึ่งในสี่เมื่อปัจจัยอื่นคงที่|เพิ่มสี่เท่า|สูตร cache เป็นสัดส่วนตรงกับจำนวน KV heads', 'เปรียบเทียบ cache ของ MHA/GQA/MQA เมื่อ T คงที่');
  add(49, 'BPE รวมคู่ที่พบบ่อยเป็น token ใหม่', 'bpe', 'จำนวน merge|0|3|1|1',
    'เริ่ม symbols~ภาพใช้ corpus banana ซ้ำสองครั้งและเริ่มจาก character symbols เพื่ออธิบาย merge|นับ adjacent pairs~คู่ a,n และ n,a พบหลายครั้ง; tie-breaking ต้อง deterministic|merge ที่เลือก~รวมคู่ที่พบบ่อยแล้วนับใหม่; merge ranks เป็น artifact ของ tokenizer|encode/decode~encode ทำตาม ranks ที่เรียนไว้; byte-level BPE ในบทเต็มรักษา UTF-8 round trip',
    'repeat: count adjacent pairs → choose max with deterministic tie → merge', 'encoding ไม่เลือกคู่ตามความถี่ของข้อความใหม่ แต่ทำตาม merge ranks ที่ train ไว้',
    'BPE encoding ใช้กฎจากไหน|merge ranks ที่เรียนตอน train|ฝึกความถี่ใหม่ทุกข้อความ|artifact tokenizer ต้องคงที่ เพื่อให้ IDs ของข้อมูลและโมเดลตรงกัน', 'เลื่อน merges แล้วดู banana สั้นลงโดยยังประกอบกลับได้');
  add(50, 'pretraining เริ่มจาก random weights', 'train', 'learning rate η|0.01|0.8|0.01|0.1',
    'เตรียม causal windows~token sequence จับ input กับ target ที่ shift หนึ่งตำแหน่ง|forward + loss~decoder เริ่มจาก random weights; cross-entropy เทียบกับ next tokens|backward + update~ภาพใช้ regression scalar แทน optimizer step เพื่ออ่าน gradient; LLM ใช้ token CE จริง|validate + checkpoint~ตรวจ held-out loss และเก็บ model, optimizer, scheduler, RNG เพื่อ resume',
    'L_LM=−mean(log P(next token)); effective batch=local batch × accumulation × world size', 'loss ของ gradient accumulation ต้อง scale ให้ตรงจำนวน microbatches/valid tokens',
    'checkpoint เพื่อ resume ต้องมีแค่ weights หรือไม่|ต้องมี optimizer และ state ที่เกี่ยวข้องด้วย|แค่ weights เพียงพอให้ trajectory เหมือนเดิม|momentum, scheduler, RNG และ data position มีผลต่อก้าวต่อไป', 'ทดลอง learning rate บน scalar surrogate แล้วอ่านการประยุกต์กับ CE ในบทเต็ม');
  add(51, 'corpus สะอาดต้องแบ่งก่อนสร้าง windows', 'data', 'สัดส่วน train (%)|50|90|5|80',
    'เก็บ provenance~เอกสารแต่ละชิ้นมี ID, source และข้อมูลที่ตรวจย้อนกลับได้|clean + dedup~เอาสำเนาซ้ำออกก่อนแบ่งเพื่อป้องกันข้อความเดียวกันข้ามชุด|split เอกสาร~ภาพแยก train/test หลัง dedup; tokenizer fit บน train แล้ว freeze|tokenize + pack~สร้าง windows ภายในชุดที่ถูกแบ่งแล้ว; เก็บ manifest และ shard ให้ทำซ้ำได้',
    'documents → dedup → split → frozen tokenizer → token windows → shards', 'split หลังทำ overlapping windows อาจทำให้ข้อความเดียวกันอยู่ทั้ง train และ validation',
    'ควร split เอกสารเมื่อไร|ก่อนสร้าง overlapping windows|หลังสุ่ม windows ทั้งหมด|windows ที่ติดกันแบ่งปัน tokens; ต้องแบ่งหน่วยต้นทางก่อน', 'ปรับ train ratio ดูเส้นทางข้อมูลหลังลบ duplicate');
  add(52, 'DDP รวม gradients ก่อน update', 'distributed', 'จำนวน workers|1|8|1|4',
    'replicate~workers ถือ model weights ชุดเดียวกันและรับข้อมูลคนละ shard|local backward~แต่ละ worker คำนวณ gradients ของ local batch|all-reduce~รวม gradients แล้ว average; ภาพแสดง scalar gradient ของแต่ละ worker|synchronized step~ทุก worker ใช้ gradient เดียวกันอัปเดต model; communication เป็น overhead',
    'g_global=(1/W)Σ_w g_w; global batch=W × local batch × accumulation', 'DDP replicate model จึงไม่ได้ลด memory ของ weights ต่อ GPU โดยตรง',
    'หลัง all-reduce ทุก worker ควรมีอะไรเหมือนกัน|gradient ที่ใช้ update|training samples ชุดเดียวกัน|ต่าง shard ได้ แต่ gradient ที่รวมแล้วทำให้ weights เดินสอดคล้องกัน', 'เพิ่ม workers ดู local gradients และค่าเฉลี่ยที่ broadcast กลับ');
  add(53, 'CUDA จัด thread ให้ตรงตำแหน่งข้อมูล', 'gpu', 'threads ต่อ block|4|16|4|8',
    'host เตรียมข้อมูล~CPU ส่ง arrays ไป device memory|launch grid~กำหนด blocks และ threads; ภาพจำลอง vector add 32 elements|map index~i=blockIdx×blockDim+threadIdx บอกว่า thread อ่าน element ใด|write + sync~แต่ละ thread เขียน c[i]=a[i]+b[i] เมื่อ i<N; host รอเมื่อต้องใช้ผล',
    'i=blockIdx.x×blockDim.x+threadIdx.x; if i<N: c[i]=a[i]+b[i]', 'GPU launch มัก asynchronous; จับเวลาโดยไม่ synchronize อาจได้แค่เวลา launch',
    'ทำไมต้องเช็ก i < N|threads บางตัวอาจเกินขนาดข้อมูล|เพื่อให้ thread ใช้ weight คนละชุด|ceil(N/block_size) อาจสร้าง threads มากกว่า N จึงต้องป้องกัน out-of-bounds', 'เปลี่ยน block size แล้วดู grouping ของ indices โดยผลรวมยังตรงเดิม');
  add(54, 'loss scaling รักษา gradient เล็ก', 'precision', 'loss scale S|1|1024|1|128',
    'forward autocast~บาง operations ใช้ precision ต่ำ; บางส่วนคง FP32 ตามความเหมาะสม|scale loss~คูณ loss ด้วย S ก่อน backward ให้ gradient เล็กขยายใน FP16|unscale~หาร gradients ด้วย S ก่อน clipping และ optimizer step|check overflow~ถ้า gradient ไม่ finite ต้องข้าม update และปรับ scale; ภาพใช้ FP16 minimum subnormal เป็นตัวอย่าง',
    'L_scaled=S L; g_scaled=S g; g=g_scaled/S', 'loss scaling ไม่ได้เพิ่ม precision ของทุก operation และ BF16 ไม่เหมือน FP16 ด้าน exponent range',
    'clip gradient ก่อนหรือหลัง unscale|หลัง unscale|ก่อน unscale เสมอ|threshold ของ clipping มีหน่วยของ gradient จริง มิฉะนั้น scale จะเปลี่ยนกติกา clip', 'เพิ่ม S ให้ gradient 10⁻⁸ พ้น minimum subnormal ของ FP16');
  add(55, 'fine-tune ต้องตรวจทั้งงานใหม่และเดิม', 'train', 'learning rate η|0.01|0.8|0.01|0.05',
    'load pretrained~เริ่มจาก weights ที่เรียนไว้แล้ว ต่างจาก pretraining from scratch|เลือก trainable~freeze ส่วนที่ต้องการและกำหนด task data ที่ไม่รั่ว evaluation|update งานใหม่~ภาพใช้ scalar objective อธิบายขนาดก้าว; fine-tuning จริงใช้ loss ของ task|evaluate before/after~ตรวจคุณภาพงานใหม่และ forgetting ของงานเดิม พร้อมเลือก checkpoint จาก validation',
    'trainable ratio=N_trainable/N_total; θ_new=θ_pretrained−ηg_task', 'trainable น้อยไม่ใช่รับประกันไม่ forget และ train loss ต่ำไม่ใช่หลักฐานว่าทักษะเดิมยังอยู่',
    'ต้องประเมินก่อน fine-tune เพื่ออะไร|มี baseline สำหรับเปรียบเทียบผลและ forgetting|เพื่อใช้ test เป็น training labels|ถ้าไม่มี baseline จะไม่รู้การเปลี่ยนแปลงจริงของโมเดล', 'ดู learning rate เล็กช่วยจำกัดการขยับ weights อย่างไร');
  add(56, 'LoRA ฝึกการเปลี่ยนแปลงขนาดเล็ก', 'lora', 'rank r|1|8|1|2',
    'freeze base~W ขนาด 8×8 คงที่; gradient ยังไหลผ่านไป adapter ได้|สร้าง A B~A ขนาด r×8 และ B ขนาด 8×r เป็น trainable matrices|เพิ่ม low-rank delta~W_eff=W+(α/r)BA; trainable count=r(d_in+d_out)|save หรือ merge~เก็บ adapter คู่กับ base รุ่นที่ตรง; QLoRA เพิ่ม quantized base แต่ compute dtype แยกต่างหาก',
    'ΔW=(α/r)BA; trainable params=r(d_in+d_out); base params=d_in d_out', 'rank สูงขึ้นเพิ่ม capacity และ memory; adapter ต้องตรงกับ architecture และ base weights',
    'rank=2 กับ matrix 8×8 มี adapter parameters เท่าไร|32|64|A มี 2×8 และ B มี 8×2 รวม 32 ไม่รวม base ที่ freeze', 'เปลี่ยน r แล้วดูขนาด A,B และสัดส่วน trainable parameters');
  add(57, 'SFT เลือก tokens ที่ต้องฝึกตอบ', 'sft', 'จำนวน assistant tokens|1|6|1|4',
    'format messages~system/user/assistant ถูก serialize ด้วย chat template ที่ตรงกับ model|tokenize~เก็บ role boundaries และ end-of-turn; ไม่ตัดคำตอบโดยไม่ตรวจ|mask labels~assistant-only loss ให้ user/system labels เป็น ignore; ภาพใช้แถบ tokens ตาม role|causal shift + CE~ทำนาย assistant token ถัดไปและเฉลี่ยเฉพาะ valid labels',
    'L=−Σ_(i assistant) log P(x_i given x_<i)/N_valid; ignored labels excluded', 'loss mask กับ attention mask ไม่ใช่อย่างเดียวกัน; user tokens ยังเป็น context แม้ไม่อยู่ใน loss',
    'user tokens ที่ไม่มี loss ยังส่งผลต่อคำตอบหรือไม่|ยังเป็น context ให้ attention|ถูกลบออกจาก input ทั้งหมด|label mask กำหนดตำแหน่งที่คิด loss ส่วน input และ attention ยังอ่าน user ได้', 'เพิ่มความยาวคำตอบ ดูจำนวน valid labels โดย user ยังอยู่ใน context');
  add(58, 'RLHF ใช้ reward และคุมการเบี่ยง', 'preference', 'KL penalty β|0|2|0.1|0.5',
    'preference pairs~มนุษย์หรือแหล่งที่มีคุณภาพเลือก preferred answer เพื่อฝึก reward model|sample policy~policy สร้างคำตอบ; reward model ให้คะแนนที่อาจมี bias|KL-shaped reward~หัก penalty เมื่อ policy ต่างจาก reference; ภาพใช้ reward และ KL ตัวอย่าง|PPO update + audit~อัปเดต policy ภายใต้ constraints และตรวจ reward hacking ด้วย evaluation อิสระ',
    'R_shaped=R_model−β KL(π || π_ref)', 'reward model score ไม่เท่ากับความพึงพอใจจริง; optimize มากเกินไปอาจ exploit reward model',
    'เพิ่ม β โดยคง reward/KL เดิมทำอะไร|ลงโทษการเบี่ยงจาก reference มากขึ้น|เพิ่ม reward ทุกคำตอบเท่ากัน|คำตอบที่ KL สูงถูกหักมากกว่า; ต้องตรวจ trade-off กับ task quality', 'เปรียบเทียบ reward ดิบและ reward หลัง KL penalty');
  add(59, 'DPO เรียนจากคู่คำตอบโดยตรง', 'preference', 'DPO β|0.1|2|0.1|0.5',
    'chosen/rejected~สองคำตอบต้องใช้ prompt เดียวกันและตรวจคุณภาพ preference|log probabilities~sum log probabilities ของ response tokens โดย mask prompt และ padding|relative margin~หัก preference margin ของ reference จาก margin ของ policy|DPO loss~−log σ(β·relative_margin) เพิ่มความได้เปรียบของ chosen แบบอิง reference',
    'z=β[(logπ(y⁺)−logπ(y⁻))−(logπ_ref(y⁺)−logπ_ref(y⁻))]; L=−logσ(z)', 'β ใน DPO ไม่ควรตีความตรง ๆ ว่าเหมือนคูณ KL penalty ในภาพ RLHF ทุกสถานการณ์',
    'ทำไมต้อง subtract reference margin|วัดการเปลี่ยน preference เทียบกับ reference|เพื่อลบ prompt ออกจาก dataset|policy ถูกเปรียบเทียบแบบ relative; reference เป็นจุดอ้างอิงของ objective', 'เปลี่ยน β แล้วดู DPO logit, probability และ loss ของคู่ตัวอย่าง');
  add(60, 'evaluation ต้องบอกความไม่แน่นอน', 'evaluation', 'จำนวนตัวอย่าง n|20|200|10|100',
    'กำหนด protocol~fix dataset, model version, prompt และ generation settings|รัน held-out~เก็บผลรายตัวอย่างและ slices; ภาพสมมติ observed accuracy=0.8|คำนวณ uncertainty~standard error ลดประมาณ 1/√n; ใช้ Wilson interval ในภาพ|เทียบและรายงาน~ตรวจ contamination, paired comparison, latency และ practical significance',
    'SE≈√(p̂(1−p̂)/n); Wilson 95% interval uses z=1.96', 'accuracy ต่างเล็กน้อยไม่ยืนยันว่ารุ่นหนึ่งดีกว่า และ confidence interval ไม่ใช่โอกาสที่คำตอบแต่ละข้อถูก',
    'เพิ่ม sample size โดย accuracy เดิม interval เป็นอย่างไร|มักแคบลง|กว้างขึ้นเสมอ|ตัวอย่างมากช่วยลดความไม่แน่นอนของ estimate ภายใต้ sampling assumptions', 'เลื่อน n แล้วดู Wilson interval หดโดย center ใกล้ 0.8');
  add(61, 'quantization ปัดค่าลงกริดตัวเลข', 'quant', 'จำนวน bits|2|8|1|4',
    'เลือก range~ตัวอย่างใช้ symmetric range [−1,1] และค่าทดสอบ w=0.37|คำนวณ scale~s=1/q_max ทำให้ integers ครอบคลุม range ที่กำหนด|round + clamp~q=clip(round(w/s)); bits น้อยมีระดับแทนค่าห่างขึ้น|dequantize + evaluate~ŵ=sq แล้ววัด error; โมเดลจริงต้องตรวจ accuracy, clipping และ kernel support',
    'q_max=2^(bits−1)−1; s=1/q_max; q=round(w/s); ŵ=sq', 'weight storage ลดไม่แปลว่า latency ลดเสมอ; KV cache และ activations ต้องคิดแยก',
    'bits น้อยทำให้ quantization levels อย่างไร|น้อยลงและระยะกริดกว้างขึ้น|มากขึ้นและละเอียดขึ้น|จำนวน integers ที่แทนได้ลดลง; rounding error อาจเพิ่มขึ้น', 'ลอง 2,4,8 bits เปรียบเทียบกริดกับ error ของ 0.37');
  add(62, 'prefill และ decode ใช้ cache ต่างกัน', 'cache', 'context tokens T|16|256|16|64',
    'prefill~อ่าน prompt หลาย tokens และสร้าง K,V ของทุกตำแหน่ง|เก็บ KV~cache บรรจุอดีตของแต่ละ layer; memory โตตาม sequence length|decode~query ของ token ใหม่อ่าน cached keys และ values ไม่ต้อง project อดีตซ้ำ|จัด memory/batch~paging และ continuous batching ลด waste; ต้องวัด TTFT, inter-token latency และ throughput',
    'KV bytes=2LBTH_kvD_head b; decode still reads prior KV', 'KV cache ไม่ได้ทำให้ decode มีต้นทุนคงที่กับทุก context length',
    'decode ยังต้องอ่านอดีตหรือไม่|ต้องอ่าน cached K,V|ไม่อ่านอดีตอีกเลย|cache ลดการคำนวณซ้ำแต่ attention ของ query ใหม่ยังอ่านข้อมูลอดีต', 'เพิ่ม T แล้วดู cache memory โตเป็นเส้นตรง');
  add(63, 'serving ต้องรับภาระให้ทัน', 'queue', 'อัตรา request λ (/s)|1|15|1|6',
    'รับ request~ตรวจขนาด prompt, budget และ timeout ก่อน admit เข้าคิว|queue + batch~scheduler เลือก requests ให้ engine ภายใต้ capacity|generate + stream~ส่ง tokens และรองรับ cancellation เพื่อคืน resources|observe + backpressure~ภาพใช้ M/M/1 สมมติ service rate μ=10/s; เมื่อ λ≥μ คิวไม่มี steady-state',
    'ρ=λ/μ; M/M/1 W=1/(μ−λ) when λ<μ; L=λW', 'mean latency ไม่แทน p95 และ queue ที่โตไม่จบแก้ด้วยเพิ่ม timeout อย่างเดียวไม่ได้',
    'เมื่อ λ เข้าใกล้ μ เวลารอเป็นอย่างไร|เพิ่มอย่างรวดเร็ว|ลดลงเพราะมี batch มาก|ใน queue model เมื่อ utilization ใกล้ 1 spare capacity ต่ำจน backlog โต', 'เพิ่ม λ เข้าใกล้ 10 แล้วดู latency และสถานะ overload');
  add(64, 'deployment มี gate ก่อนรับ traffic', 'release', 'ขั้น rollout|0|4|1|2',
    'build artifact~pack model/runtime/config เป็น version ที่ตรวจย้อนกลับได้|start process~liveness ตรวจว่า process ยังอยู่; ไม่ยืนยันโมเดลพร้อมรับงาน|readiness gate~load model และ warmup สำเร็จก่อนส่ง traffic|rollout + rollback~ค่อย ๆ เปลี่ยน version พร้อม metrics; ถ้า health/quality fail ต้องย้อนกลับได้',
    'build → start → readiness → canary → promote/rollback', 'process ที่ตอบ health ได้อาจยังโหลด model ไม่เสร็จ; liveness กับ readiness มีหน้าที่ต่างกัน',
    'ควรส่ง traffic เมื่อใด|หลัง readiness ผ่าน|ทันทีที่ container เริ่ม|readiness ยืนยัน service พร้อมทำงานที่ต้องรับจริง', 'เลื่อน rollout ดูจังหวะที่ traffic เปิดได้หลัง readiness');
  add(65, 'MLOps เชื่อม version กับการเฝ้าดู', 'monitor', 'การเปลี่ยน mean ของข้อมูล|0|3|0.1|1',
    'track run~เชื่อม data, code, config, metrics และ artifact ต่อ run ID|registry + gate~promotion ต้องผ่าน quality/compatibility checks ไม่ใช่เลือก train loss ต่ำสุด|serve version~ผูก deployment กับ model version เพื่อ trace prediction กลับได้|monitor drift~ภาพเลื่อน input distribution ใหม่เทียบ reference; drift เป็นเหตุให้ตรวจ ไม่ใช่ข้อพิสูจน์ quality ลด',
    'data drift: P_new(x) ≠ P_ref(x); concept drift: P(y given x) changes', 'data drift ไม่ยืนยัน concept drift และ prediction distribution เปลี่ยนไม่ได้บอกสาเหตุเดียว',
    'พบ data drift ควรสรุปว่าโมเดลผิดทันทีหรือไม่|ต้องตรวจ quality และ slices เพิ่ม|โมเดลผิดทุก prediction แน่นอน|distribution เปลี่ยนเป็นสัญญาณตรวจสอบ; ต้องดู labels และ performance จริง', 'เลื่อน mean ดู distribution สองช่วงแยกกันมากขึ้น');
  add(66, 'data contracts กันข้อมูลผิดก่อนเข้า AI', 'data', 'สัดส่วน train ในตัวอย่าง (%)|50|90|5|70',
    'ingest~รับ records พร้อม source และ event metadata|validate~ตรวจ schema, missing และ duplicates; invalid records ไม่ควรผ่านเงียบ ๆ|transform~ภาพใช้ clean/split ตัวอย่างเล็ก; งานจริงมี ETL/ELT, partition และ event time|publish version~เก็บ manifest และ lineage ให้ consumers อ่านชุดที่ตรวจแล้วได้ซ้ำ',
    'source → validation → transformation → versioned dataset → consumer', 'processing time ไม่เท่ากับ event time; retry ingest ต้องระวัง duplicate side effects',
    'data contract ตรวจอะไร|schema และกติกาคุณภาพที่ตกลงกับ consumer|แค่ชื่อไฟล์|ข้อมูลชนิดผิดหรือ feature หายทำให้ model ใช้งานไม่ได้แม้ pipeline รันสำเร็จ', 'ติดตาม record ที่ถูกตัดออกและชุดที่ publish หลัง validation');
  add(67, 'distributed systems ต้องมี backpressure', 'queue', 'อัตรางานเข้า λ (/s)|1|15|1|6',
    'schedule~scheduler แจกงานให้ workers; node บางตัวอาจช้าหรือหาย|retry~failure ชั่วคราวใช้ bounded retries และ backoff; idempotency กันผลซ้ำ|control queue~ภาพใช้ queue model เพื่อเห็น backlog เมื่อ input เร็วกว่าการประมวลผล|stop overload~bounded admission และ circuit breaker ช่วยไม่ให้ failure ลุกลาม',
    'ρ=arrival_rate/service_rate; retry budget bounded; effects idempotent', 'at-least-once delivery อาจเรียกงานหลายครั้ง; exactly-once effect ต้องออกแบบ state/transaction ด้วย',
    'ทำไม retry ต้องมี idempotency|ป้องกัน side effects ซ้ำ|ทำให้ network ไม่มีวันล้ม|request อาจสำเร็จแล้วแต่ response หาย การเรียกซ้ำจึงต้องไม่ทำผลเดิมสองครั้ง', 'เพิ่ม λ แล้วดูเมื่อ spare capacity หายไป');
  add(68, 'guardrails มีหลายชั้นตรวจคนละเรื่อง', 'agent', 'งบตรวจ/ลงมือเป็นขั้น|1|6|1|4',
    'validate input~กำหนดขนาด รูปแบบ และขอบเขต trust ของข้อมูลภายนอก|authorize context/tool~สิทธิ์ document และ tool ต้องตรวจใน application ไม่ฝากให้ prompt|execute constrained~schema, allowlist และ resource budget จำกัด actions ที่ executor ทำได้|validate output + audit~ตรวจรูปแบบ/การใช้ผลลัพธ์และเก็บ traces; ภาพเป็นลำดับ gate จำลอง',
    'input checks → context ACL → tool authorization → output validation → audit', 'system prompt ไม่ใช่ security boundary และ retrieval content อาจมี prompt injection',
    'prompt บอกให้เข้าถึงไฟล์ได้ ถือว่าอนุญาตแล้วหรือไม่|ต้องตรวจสิทธิ์ใน application|ถือว่า model อนุมัติได้เอง|authorization ต้องบังคับจากระบบที่คุม resource ไม่ใช่ข้อความที่อาจถูกโจมตี', 'เลื่อน budget ดู gate ที่ยังไม่ผ่านก่อน executor ลงมือ');
  add(69, 'attribution แยกส่วนร่วมของ features', 'attribution', 'ค่า feature x₁|0|3|0.1|1',
    'fix prediction~ภาพใช้ linear model f(x)=2x₁−x₂+0.5 โดย x₂=1|เลือก baseline~baseline x=(0,0) ให้ f=0.5; attribution ขึ้นกับ baseline|คำนวณ contribution~linear contributions คือ wᵢ(xᵢ−baselineᵢ)|ตรวจผลรวม~ผล contributions บวก baseline ต้องตรง prediction; model explanation ไม่ใช่ causal claim',
    'f(x)=0.5 + 2x₁ − x₂; Σ contributions=f(x)−f(baseline)', 'feature attribution สูงไม่แปลว่าการเปลี่ยน feature ในโลกจริงทำให้ outcome เปลี่ยนตามนั้นเสมอ',
    'ผลรวม attribution อธิบายอะไรในตัวอย่าง|prediction ต่างจาก baseline เท่าไร|สาเหตุจริงของ outcome เสมอ|completeness เป็นความสัมพันธ์ใน model กับ baseline ที่กำหนด ไม่พิสูจน์ causality', 'เพิ่ม x₁ ดู contribution บวกและ prediction เปลี่ยน');
  add(70, 'compression ต้องวัดต้นทุนจริง', 'compression', 'สัดส่วน pruning (%)|0|90|10|40',
    'เลือกเป้าหมาย~storage, RAM, FLOPs และ latency เป็นคนละ budget|prune weights~ภาพตัด weights ขนาดเล็กก่อนแล้วแสดง mask ของ matrix 8×8|recover quality~fine-tune หรือ distill จาก teacher เพื่อชดเชยพฤติกรรมที่เสียไป|export + benchmark~unstructured zeros ยังต้อง sparse representation/kernel จึงประหยัดจริง; วัด accuracy ด้วย',
    'sparsity=N_zero/N_total; dense storage unchanged by zero mask alone', 'mask weights เป็นศูนย์ไม่ได้ทำให้ dense matrix ใช้ bytes น้อยลงโดยอัตโนมัติ',
    'prune 50% แล้ว dense tensor ใช้ memory ลดครึ่งหรือไม่|ไม่ลดถ้ายังเก็บ dense แบบเดิม|ลดทันทีเสมอ|ต้องเปลี่ยน representation หรือโครงสร้างและใช้ kernel ที่รองรับเพื่อประหยัดจริง', 'เพิ่ม pruning ดู zeros มากขึ้น แต่ dense bytes ยังเท่าเดิม');
  add(71, 'MoE เลือก experts ต่อ token', 'experts', 'จำนวน experts ที่ activate k|1|4|1|2',
    'router scores~router อ่าน hidden state ของ token และให้คะแนนแต่ละ expert|top-k~เลือก experts ที่ score สูงสุด; ภาพใช้ logits [2,1,0,−1]|dispatch~ส่ง token ไป expert ที่เลือกและตรวจ capacity/load|weighted combine~normalize weights ของ experts ที่เลือกแล้วรวม output; ตรวจ load balancing',
    'y=Σ_(e in top-k) softmax(selected router logits)_e E_e(x)', 'total parameters ไม่เท่ากับ active parameters; k น้อยยังมี routing/communication cost',
    'activate 2 จาก 4 experts แปลว่าอะไร|คำนวณ expert FFN สองตัวต่อ token|เหลือ weights ใน storage แค่สองตัว|ยังเก็บ experts ทั้งหมด แต่ใช้เฉพาะ subset ต่อ token', 'เปลี่ยน k ดู experts ที่เปิดและน้ำหนักหลัง normalize');
  add(72, 'context window คุมสิ่งที่อ่านได้', 'attention', 'sliding window size|1|4|1|2',
    'รับ sequence~context tokens เป็นข้อมูลที่ model เห็น ณ request นั้น ไม่ใช่ persistent memory|จำกัด attention~ภาพแสดง causal sliding window อ่านล่าสุด W tokens|จัด KV/context~discard นอก window ลด cache ของรูปแบบนี้; external retrieval เป็นอีกกลไก|ประเมินระยะไกล~งานที่ต้องใช้ข้อมูลเก่าอาจเสียคุณภาพ แม้ memory ลดลง',
    'allowed keys: max(0,i−W+1) ≤ j ≤ i; dense attention O(T²)', 'long context ไม่รับประกัน recall ของข้อมูลทุกตำแหน่ง และ KV cache ไม่ใช่ vector database',
    'window=1 query อ่านอะไรได้|ตำแหน่งตัวเองเท่านั้น|ทุกตำแหน่งในอดีต|ขอบซ้าย i−W+1 เท่ากับ i จึงเหลือ key เดียว', 'เพิ่ม W ดูพื้นที่ attention ที่อนุญาตกว้างขึ้น');
  add(73, 'reasoning ใช้ compute กับการตรวจ', 'reasoning', 'จำนวน candidates N|1|6|1|3',
    'ตั้งปัญหา~กำหนด budget และสิ่งที่ตรวจคำตอบได้|สร้าง candidates~ภาพใช้คำตอบจำลองหลายทาง ไม่ใช่ chain of thought จากโมเดลจริง|verify~ตรวจด้วยกติกา deterministic หรือ verifier ที่ประเมินแล้ว; votes ยังอาจผิดร่วมกัน|เลือกและรายงาน~เลือก candidate ผ่านเกณฑ์ภายใน budget; วัดความถูกต้องกับ latency/cost',
    'test-time compute ∝ candidate count × cost per candidate (illustrative)', 'คำอธิบายยาวหรือเสียงส่วนใหญ่ไม่ได้ยืนยันคำตอบถูก และ verifier อาจถูก exploit',
    'เพิ่มจำนวน candidates รับประกันถูกหรือไม่|ไม่ ถ้าพลาดร่วมกันหรือ verifier ผิด|รับประกันเมื่อ N มากกว่า 3|error ที่สัมพันธ์กันและเกณฑ์ตรวจผิดทำให้ search ไม่ช่วยเสมอ', 'เพิ่ม N ดู candidates ที่ผ่าน verifier และต้นทุนสมมติ');
  add(74, 'VLM ใช้ visual tokens ใน language context', 'patch', 'ขนาด image patch|2|8|2|4',
    'vision encode~ภาพแปลงเป็น patch features จาก encoder|project~projector จัด dimension/representation ให้ language model อ่านได้|fuse context~visual tokens และข้อความรวมด้วย token injection หรือ cross-attention|generate + grounding~ตอบตามภาพแล้วตรวจ hallucination/OCR/ตำแหน่ง; ภาพสอนผล patch count ต่อ context budget',
    'visual tokens≈(H/P)(W/P); projector: D_vision → D_language', 'dimension ตรงไม่ได้แปลว่าโมเดลเข้าใจภาพ; ต้องเรียน multimodal alignment',
    'เพิ่ม resolution โดย patch size เดิมเกิดอะไร|visual tokens เพิ่มและใช้ context มากขึ้น|token count คงที่เสมอ|พื้นที่ภาพมากขึ้นทำให้จำนวน patches เพิ่มในรูปแบบ patchification นี้', 'ลด P ดู visual tokens เพิ่มและต้นทุน attention สูงขึ้น');
  add(75, 'voice streaming ต้องเลือก chunk ที่พอดี', 'audio', 'audio chunk size (samples)|8|32|8|16',
    'รับเสียงเป็น chunks~chunk เล็กส่งต่อเร็วแต่ overhead สูง; ภาพใช้ waveform ที่สร้างไว้|encode audio~ใช้ continuous features หรือ neural codec tokens ตาม architecture|เข้าใจและตอบ~cascaded ASR→LLM→TTS หรือ native audio model มี trade-off ต่างกัน|turn-taking~endpointing และ barge-in คุมการจบ/แทรก; latency ของทั้ง pipeline ต้องรวมกัน',
    'chunk duration=N/sample_rate; cascade latency≈ASR+LLM+TTS+buffering', 'หยุดพูดชั่วคราวไม่ใช่จบ turn เสมอ; chunk เล็กไม่รับประกัน end-to-end latency ต่ำที่สุด',
    'ทำไมต้องรวม buffering ใน latency|ผู้ใช้รอเก็บ chunk ก่อนประมวลผลด้วย|เพราะ model ไม่ต้องคำนวณ|เวลารอ input และ scheduling เป็นส่วนของเวลาที่ผู้ใช้รับรู้', 'เพิ่ม chunk ดูระยะเวลาที่ต้องเก็บก่อนส่งต่อ');
  add(76, 'world model ทำนายหลายก้าวแล้วผิดสะสม', 'world', 'planning horizon|1|8|1|4',
    'encode state~observation เปลี่ยนเป็น state/latent ที่เหมาะกับ dynamics|predict next~ตัวอย่างจริง x_next=x+a; learned model มี bias +0.1 ต่อก้าว|rollout candidate~ลอง action sequence ใน model; prediction error สะสมตาม horizon|plan และ execute~เลือกแผนแล้วทำเพียงบางก้าวก่อนรับ observation ใหม่; model uncertainty ต้องนำมาคิด',
    'true: x_(t+1)=x_t+a; toy model: x̂_(t+1)=x̂_t+a+0.1', 'world model ที่ดูสมจริงอาจทำนาย dynamics ที่ใช้วางแผนไม่ถูก และ planner อาจ exploit model error',
    'rollout ยาวขึ้นกับ model ที่มี bias คงที่เกิดอะไร|error สะสมหลายก้าว|error หายไปเอง|prediction ใหม่ใช้ state ที่ผิดจากก้าวก่อน ทำให้ bias บวกต่อกันในตัวอย่าง', 'เพิ่ม horizon เปรียบเทียบ true กับ predicted trajectory');
  add(77, 'robot ใช้ feedback กลับเข้าวงจร', 'robot', 'controller gain Kp|0.1|1.8|0.1|0.5',
    'sense~วัดตำแหน่งจริง; state estimate อาจมี noise และ delay|compute error~target=1 ลบ current position x เพื่อกำหนดทิศทาง|act~ตัวอย่าง discrete controller x_next=x+Kp(target−x); ระบบจริงมี limits/dynamics|feedback ซ้ำ~วัดใหม่ทุกก้าว; gain สูงอาจ overshoot และ oscillate',
    'e_t=target−x_t; x_(t+1)=x_t+Kp e_t [toy unit-step dynamics]', 'คำตอบของ policy ยังต้องผ่าน action limits และ safety constraints ของ robot จริง',
    'เหตุใดต้องวัด state ใหม่หลัง act|เพราะผลจริงอาจไม่ตรงที่คาด|เพราะ target เปลี่ยนทุกครั้งเสมอ|closed-loop control แก้ error จากผลจริง; open-loop ไม่มี feedback นี้', 'ลอง Kp=0.5 กับ 1.5 ดูการเข้าเป้าและ overshoot');
  add(78, 'TinyML ต้องผ่าน storage และ RAM', 'edge', 'precision ของ weights (bits)|4|32|4|8',
    'คำนวณ weights~ตัวอย่างมี 100,000 parameters; precision กำหนด model storage ขั้นต่ำ|คิด runtime RAM~activations, temporary buffers และ runtime ใช้ RAM แยกจาก storage|ตรวจ limits~ภาพใช้ flash 300 KB กับ RAM budget สมมติ; ต้องตรวจ operators ด้วย|run บน device~วัด latency/energy/quality จริง ไม่ใช้ desktop benchmark แทน',
    'weight bytes=N_params × bits/8; peak RAM≠weight storage alone', 'weights ใส่ flash ได้ยังไม่ยืนยัน inference ใช้ RAM พอ และ 4-bit ต้องมี kernel รองรับ',
    'weights 100k ที่ 8 bits ใช้ bytes ขั้นต่ำเท่าไร|100,000 bytes|800,000 bytes|8 bits คือ 1 byte ต่อ parameter ยังไม่รวม metadata/scales/runtime', 'เปลี่ยน bits ดู storage เทียบ flash limit โดย RAM จำลองยังแยกต่างหาก');
  add(79, 'paper claim ต้องผ่านการทดลองที่ควบคุม', 'evaluation', 'จำนวนทดลอง/ตัวอย่าง n|20|200|10|100',
    'แปลง claim~นิยาม metric, dataset, baseline, compute และ expected result ที่ทดสอบได้|implement specification~เชื่อมสมการกับ code พร้อม shapes และ invariants|controlled experiment~เปลี่ยนตัวแปรเดียว ใช้ multiple seeds/paired results ตามโจทย์|ตีความ evidence~ภาพใช้ Wilson interval ของ accuracy เป็นตัวอย่าง uncertainty; รายงาน reproduction gap และข้อจำกัด',
    'effect=metric_method−metric_baseline; uncertainty depends on protocol and sample size', 'ตัวเลขใกล้ paper ไม่พิสูจน์ว่าทำซ้ำกลไกถูก และ baseline ที่ compute ต่างกันอาจเปรียบเทียบไม่ยุติธรรม',
    'ablation ควรเปลี่ยนอะไร|ส่วนที่ต้องการทดสอบพร้อมคุมปัจจัยอื่น|ทุกส่วนพร้อมกัน|หากเปลี่ยนหลายอย่างจะระบุไม่ได้ว่าส่วนใดทำให้ผลต่าง', 'เพิ่ม n ดู uncertainty แล้วถามว่าช่วงที่ต่างยังมีความหมายกับ claim หรือไม่');
  add(80, 'ประกอบ LLM ตั้งแต่ corpus ถึง service', 'capstone', 'ขั้น lifecycle|0|7|1|3',
    'corpus → tokenizer~clean/dedup/split ก่อน fit tokenizer จาก train; เก็บ data manifest|decoder → pretrain~initialize random weights แล้ว causal CE, validate/checkpoint; ห้ามใช้ pretrained เป็น primary model|SFT → preferences~format/mask assistant labels แล้วทดสอบ alignment แบบมี held-out protocol|quantize → serve → monitor~ตรวจ memory/quality ก่อน export, API, deployment และ drift; local smoke ไม่ใช่ production proof',
    'corpus → tokenizer → random-init LM → pretrain → evaluate → SFT/DPO → quantize → serve/monitor', 'capstone ผ่านเมื่อ lifecycle ตรวจย้อนกลับได้ ไม่ใช่แค่ generation ออกเป็นข้อความ',
    'primary model ใน capstone เริ่มจากอะไร|random initialization|pretrained checkpoint ที่โหลดมา|เป้าหมายบทนี้คือประกอบและเข้าใจการฝึกตั้งแต่ต้น จึงกำหนด random-init เป็น primary', 'เลื่อน lifecycle แล้วอธิบาย artifact ที่ต้องได้ก่อนส่งต่อขั้นถัดไป');
  window.VISUAL_LESSONS = lessons;
})();
