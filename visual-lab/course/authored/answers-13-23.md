# เฉลยเฉพาะข้อ: classical ML และ framework
## 13
1. margin เชิงเรขาคณิตของจุดคือ y(wᵀx+b)/‖w‖; w=(2,0), b=0, x=(1,3), y=1 ได้ 1 หน่วย ไม่ใช่ raw score 2
2. hyperplane คือ {x:wᵀx+b=0}; w=(1,1), b=−2 ให้เส้น x₁+x₂=2 ในสองมิติ จุด (1,1) อยู่บนเส้น
3. hinge ℓ=max(0,1−yf); y=1,f=.3 ได้ .7; f=2 ได้ 0 จึงไม่เพิ่มแรงผลักเมื่อ functional margin ถึง 1
4. soft margin ยอม slack ξ≥0 โดยแก้ ½‖w‖²+CΣξ ภายใต้ yf≥1−ξ; จุดผิดไม่ทำให้ปัญหา infeasible เหมือน hard margin
5. hyperplane ระบุตำแหน่ง boundary ส่วน margin วัดระยะจาก boundary; คูณ w,b ด้วย 2 ไม่เปลี่ยนทั้ง boundary และ geometric margin
6. C สูงเพิ่มต้นทุนของการละเมิด margin จึงอาจตาม outlier มากขึ้น; C ต่ำยอม training errors เพื่อให้ ‖w‖ เล็ก ห้ามสรุปว่า accuracy test เพิ่มเสมอ
7. kernel trick แทน inner product ด้วย K แต่ Gram matrix ยังขนาด n×n; n=100000 แบบ float64 ใช้ 80 GB เฉพาะ matrix จึงอาจต้อง LinearSVC/approximation
8. RBF exp(−γ‖x−z‖²): ระยะ²=4, γ=.5 ได้ exp(−2)=.135335; γ สูงทำให้ locality แคบและเสี่ยงจำ noise ต้องเลือกพร้อม C บน validation
9. ใช้ `f=X@w+b; margins=y*f/np.linalg.norm(w)`; X=[[1,0],[-1,0]], y=[1,-1], w=[2,0], b=0 ได้ [1,1]; w=0 ต้อง reject
10. ตรวจ [1,1] ด้วย assert_allclose; พลิก y ตัวแรกแล้ว margin เป็น −1; rescale w,b ด้วยค่าบวก margin ต้องเท่าเดิม และ w=0 ต้อง raise ValueError
11. `loss=np.maximum(0,1-y*(X@w+b)).mean()`; f=[.3,2], y=[1,1] ให้ .35; gradient active mask y*f<1 เท่านั้น จุด kink ต้องระบุนิยาม subgradient
12. สร้าง make_circles seed=7 แบ่ง train/val ก่อน StandardScaler เปรียบเทียบ linear SVC กับ RBF C=1,γ=.1/1; บันทึก margin violations, balanced accuracy และ #SV ผลต้องวัดจาก run ไม่แต่งคะแนน
13. X เป็น (n,d), w เป็น (d,), y เป็น (n,) มีค่า ±1; y(n,1) คูณ f(n,) จะ broadcast เป็น (n,n) ผิด ต้อง reshape และ assert loss vector (n,)
14. RBF ระยะ²อาจติดลบเล็กน้อยจาก cancellation ให้ clip ≥0; standardize ก่อน และไม่ invert Gram matrix โดยตรงเมื่อจุดซ้ำ
15. เลือก scaler,C,γ จาก train/validation หรือ inner CV เท่านั้น; test เปิดหลัง freeze; กลุ่มผู้ป่วย/เวลาเดียวกันต้องไม่ข้าม split
16. วัด `perf_counter` และ peak memory ตาม n=100,200,400; linear score O(nd), kernel storage O(n²), training solver ขึ้นกับการลู่เข้า จึงไม่อ้าง cubic เป็นเวลาตายตัว
17. X→scale→f=Xw+b→active=max(0,1−yf)→objective→subgradient→update; binary predict sign(f); kernel score ΣαᵢyᵢK(xᵢ,x)+b ใช้เฉพาะ SV
18. ใช้ X,y,C,loss reduction เดียวกัน; `SVC(kernel='linear')` กับ scratch ต้องเทียบ decision scores หลังลู่เข้าและ tolerance ไม่เทียบกับ squared-hinge LinearSVC โดยไม่ปรับ objective
19. เปรียบเทียบ linear/RBF บน circles โดยคง split,scale,C; รายงาน val balanced accuracy,#SV,เวลา พร้อม seed หลายค่า; ยอมรับผลไม่ดีขึ้นหากมีหลักฐาน
20. บันทึก seed/split/scaler/C/γ/package/commit; monitor score margin และ class drift; สมมติฐาน “RBF เพิ่ม recall บน nonlinear holdout ≥.05 ที่ latency≤2×” ต้องทดสอบกับ linear baseline
## 14
1. K-Means หา k centroid เพื่อลด Σᵢ‖xᵢ−μcᵢ‖²; input X(n,d), output labels(n,),centroids(k,d); label เป็นกลุ่ม ไม่ใช่คลาสที่รู้ล่วงหน้า
2. centroid คือค่าเฉลี่ยเวกเตอร์; จุด (0,0),(2,0),(4,3) มี centroid (2,1) ไม่จำเป็นต้องตรงกับจุดใดในชุด
3. inertia คือผลรวม squared distances ถึง centroid ที่ถูก assign; X=[0,2,10,12], μ=[1,11] ได้ 1+1+1+1=4
4. DBSCAN ใช้ ε-neighborhood และ min_samples รวมตัวเอง; core ขยาย connectivity, border เชื่อม core แต่ไม่ขยายต่อ, noise ไม่ถึง core ใด ไม่มีการโหวต label แบบ KNN
5. K-Means คือกระบวนการ assign/update ส่วน centroid คือ parameter ของกลุ่ม; เปลี่ยนจุดหนึ่งมีผลต่อค่าเฉลี่ยและอาจเปลี่ยน assignment รอบถัดไป
6. ε=.35,min_samples=3 กับจุดห้องทดลองให้ 2 กลุ่ม,6 core,1 border,1 noise; เพิ่ม ε อาจเชื่อมกลุ่มและลด noise แต่ไม่รับประกัน semantic ถูก
7. hierarchical clustering แบบ agglomerative รวมแล้วแยกคืนระหว่าง fit ไม่ได้; single linkage เกิด chaining ได้ ให้ดู dendrogram และลอง complete linkage บน bridge points
8. silhouette s=(b−a)/max(a,b) วัด within-vs-nearest-other distance; ต้องมี 2≤k<n และ distance เหมาะสม คะแนนสูงไม่รับประกันว่า noise/คลาสเชิงธุรกิจถูก
9. X=[0,2,10,12], เริ่ม μ=[0,10]: assign=[0,0,1,1], update=[1,11], inertia=4; ทำซ้ำจน labels คงที่ และกำหนดนโยบาย empty cluster
10. assert centroids [1,11], inertia4; permutation cluster IDs ต้องเทียบ partition ไม่เทียบเลข label; k>n reject และจุดเหมือนกันไม่ควรสร้าง NaN
11. `np.square(X-centroids[labels]).sum()`; X(4,1),centroids(2,1),labels[0,0,1,1] ได้4 ต้อง sum ทุก feature และทุก sample ไม่ใช่ mean
12. ใช้จุด DBSCAN ใน lab seed7; sweep ε=.25/.35/.6 และ min_samples3; บันทึก core/border/noise และ silhouette เฉพาะเมื่อ valid อย่าแทน noise เป็นอีกคลัสเตอร์โดยไม่อธิบาย
13. ต้อง broadcast X[:,None,:]−centroids[None,:,:] เป็น (n,k,d) ก่อน sum(axis=2)→(n,k); sum axis ผิดทำให้ assign feature แทน cluster
14. คำนวณระยะหลัง scale และใช้ float64 บน tiny oracle; empty cluster mean ให้ NaN ต้อง reinitialize ด้วยจุดไกลที่สุดหรือตรึงค่าเดิมตามนโยบาย
15. fit scaler/centroid บน train แล้วประเมิน holdout distances; เลือก k/ε ด้วย validation+ความหมาย domain; DBSCAN ไม่มี predict มาตรฐานจึงต้องระบุนโยบายสำหรับจุดใหม่
16. Lloyd หนึ่งรอบ O(nkd) เวลา, distance tensor O(nkd) หาก materialize; chunk ลด memory; naive DBSCAN O(n²) ระยะ/เวลาในกรณีเลวร้าย ต้องวัด index ที่ใช้จริง
17. K-Means: initialize→distance→argmin→mean→repeat; DBSCAN: neighborhoods→core mask→core connectivity→border→noise; ห้ามใส่ centroid update ใน DBSCAN
18. ใช้ sklearn KMeans(init=centroids,n_init=1,algorithm='lloyd') เทียบ inertia และ matched centroids; DBSCAN เทียบ core mask/partition โดยยอม cluster ID permutation
19. คง X,scale,distance เปลี่ยน single→complete linkage; bridge points ควรทำให้โครงสร้าง dendrogram ต่าง ตรวจ cophenetic distances และ silhouette แทนสมมติว่าต้องดีขึ้น
20. ผ่านเมื่อบันทึก X/scale/metric/k/ε/min_samples/seed และตรวจความหมายกลุ่มกับ domain; ไม่ผ่านถ้าอ้าง cluster ID เป็น label จริง; ทดลอง “complete linkage ลด chaining บน bridge data” เทียบ linkage เดิม
## 15
1. PCA หาแกน orthonormal ที่อธิบาย variance สูงสุดของข้อมูลที่ center แล้ว; X(n,d)→Z(n,k),k≤min(n,d) ไม่ใช้ y ในการ fit
2. sample covariance XcᵀXc/(n−1); X=[[1,2],[3,4],[5,6]],mean=[3,4] ได้ [[4,4],[4,4]] และ denominator=2
3. eigenvector v ทำให้ Cv=λv; C=[[4,4],[4,4]] ให้ v₁=(1,1)/√2,λ₁=8 และ v₂=(1,−1)/√2,λ₂=0 เครื่องหมาย v กลับได้
4. explained variance ratio λᵢ/Σλ; ตัวอย่าง λ=[8,0] เก็บหนึ่งแกนได้1 หรือ100%; variance สูงไม่เท่ากับ label information สูง
5. covariance เป็น matrix ความแปรปรวนร่วมที่ใช้หาแกน ส่วน PCA รวม centering,eigendecomposition,projection และ reconstruction; covariance เพียงอย่างเดียวยังไม่มี reduced coordinates
6. เลือก k จาก cumulative EVR บน train; เกณฑ์ .95 อาจเก็บ3แกนแทน2 จึงเพิ่ม memory/เวลา แต่ไม่รับประกัน classification ดีขึ้น
7. t-SNE รักษา local neighborhoods เป็นหลัก ระยะระหว่างกลุ่มและขนาดกลุ่มใน2Dไม่ควรตีความเป็นระยะ/ความหนาแน่นจริง; เปลี่ยน seed/perplexity ตรวจความคงตัว
8. UMAP n_neighbors เล็กเน้น local ส่วนใหญ่เน้นโครงสร้างกว้าง; min_dist ควบคุมความแน่นใน embedding; รูปสวยยังต้องวัด trustworthiness และ downstream holdout
9. `mean=X.mean(0); U,S,Vt=np.linalg.svd(X-mean,full_matrices=False); Z=(X-mean)@Vt[:1].T`; ตัวอย่างข้อ2ได้พิกัด [−2√2,0,2√2] ถึง sign
10. assert Z.shape==(3,1),components@components.T≈I, centered.mean≈0, reconstruction≈X; ใช้ projector VᵀV เทียบเมื่อ sign/degenerate eigenbasis ต่าง
11. `vals,vecs=np.linalg.eigh(C); order=np.argsort(vals)[::-1]`; C ข้อ2ต้อง λ≈[8,0]; ห้าม eigh กับ covariance ที่ไม่ symmetric
12. seed7 สร้างข้อมูล anisotropic; fit PCA train เปรียบเทียบ k1/2 EVR,reconstruction MSE; UMAP เป็น optional integration ต้องติดตั้งและวัด trustworthiness จริงแยกจาก PCA pass
13. components sklearn เป็น (k,d), projection (X−μ)@components.T→(n,k); ลืม transpose อาจ fail หรือเงียบเมื่อ k=d ต้อง assert axis semantics
14. eigendecomposition ของ XᵀX ทำให้ condition number กำลังสอง; ใช้ SVD โดยตรง; variance ศูนย์ทำ EVR 0/0 ต้องกำหนดผลหรือแจ้ง error
15. split ก่อน fit center/scaler/PCA แล้ว transform holdout ด้วย train parameters; fit t-SNE บน test ร่วมแล้วใช้ภาพเลือกโมเดลทำให้การประเมินปนเปื้อน
16. dense SVD O(nd·min(n,d)), covariance O(nd²) และ O(d²) memory; วัดตาม n,d; randomized SVD ใช้เมื่อ kเล็กพร้อมตรวจ reconstructionerror
17. X→trainmean→Xc→SVD→top components→Z→ZV+mean; reconstruction error เท่ากับพลังงาน singular values ที่ตัดทิ้งหารจำนวน element ตาม reduction
18. sklearn PCA(svd_solver='full') กับ scratch เทียบ EVR,projection projector,reconstruction; align signs ก่อนเทียบ coordinates; center/scaling/ddof ต้องตรง
19. คง data/seed/iterations เปลี่ยน perplexity5→30 โดย n>30; วัด neighborhood preservation ไม่ใช่เลือกภาพที่แยกกลุ่มสวยที่สุดแล้วอ้างความจริง
20. บันทึก mean/scaler/components/solver/seed/version; monitor projected drift และ reconstructionerror; ทดลอง “PCA95%ลดเวลาclassifierอย่างน้อย20%โดยaccuracyลด≤1จุด” ด้วย holdout เดิม
## 16
1. z-score z=(x−μ)/σ บอกระยะจากค่าเฉลี่ยเป็นหน่วยส่วนเบี่ยงเบน; μ=10,σ=2,x=16 ได้3; σ=0 ต้องกำหนดนโยบาย
2. Isolation Forest สุ่ม feature/split เพื่อแยกจุด; จุดโดดเดี่ยวใช้ path สั้นกว่า จึง anomaly score สูงตาม normalization ไม่ใช่ distance ถึง centroid
3. One-Class SVM หา boundary ของ train normal ใน feature space; f(x)=ΣαᵢK(xᵢ,x)−ρ ค่าลบจัดนอก boundary โดย ν/γ ต้องเลือกบน validation
4. LOF เปรียบเทียบ local reachability density ของจุดกับเพื่อนบ้าน; density ต่ำกว่าเพื่อนมากให้ LOF>1 จึงจับ local anomaly ในกลุ่มความหนาแน่นต่างกันได้
5. z-score มองรายแกนและอิง mean/std จึงพลาด correlated anomaly; IF สุ่ม partition หลายfeature จับ nonlinear pattern ได้ แต่ score ยังขึ้นกับ train distribution
6. n_neighbors เล็กจับ local noise มากขึ้น; ใหญ่กลายเป็นมุมมอง global; LOF sklearn novelty=False ใช้ fit_predict train ส่วนจุดใหม่ต้อง novelty=True ตั้งแต่สร้าง
7. threshold ไม่ใช่ probability calibrated; ตั้ง percentile ให้ alert5%ไม่ได้แปลว่า5%เป็น anomaly จริง ต้องตรวจ false positives และต้นทุน missed event
8. precision ลดมักแลก recall เมื่อ threshold ผ่อน; prevalence ต่ำทำ PRสำคัญ เช่น TP8,FP92,FN2 ได้ precision.08,recall.8 แม้จับได้เกือบหมด
9. `mu=X.mean(0); sd=X.std(0); z=np.divide(X-mu,sd,out=np.zeros_like(X,dtype=float),where=sd>0)`; X=[1,2,3] ได้ [−1.2247449,0,1.2247449]
10. assert z mean≈0,std≈1 สำหรับ nonconstant feature; X=[2,2] ต้อง finite zeros; emptyinput reject; tiny zเทียบ population ddof0 ตามนิยามข้อ9
11. ใช้ RBF score `K=np.exp(-gamma*((X[:,None]-SV[None,:])**2).sum(-1)); f=K@alpha-rho`; SV=[[0]],α=[1],γ=1,ρ=.5 ให้ f(0)=.5,f(2)=−.481684
12. seed7 สร้างสองnormalclustersกับ outliers; fit IF/LOF บน train normal เท่านั้น; คง testlabelsเพื่อPR-AUC/recall@budget และรายงานthresholdจากvalidation
13. sklearn score_samples บางโมเดลยิ่งต่ำยิ่งผิดปกติ ต่างจาก anomaly scoreยิ่งสูง; ทำทิศทางเดียวก่อน ROC/PR อย่ากลับ labels เพื่อแต่งคะแนน
14. σเล็กทำ zใหญ่และRBF underflowได้; standardizeด้วยtrainstats,guard σ=0,ใช้ finitechecks; distance overflowใช้float64/scaleและไม่ silentlyclip inputผิด
15. novelty detection trainเฉพาะnormalที่เชื่อถือได้, validationเลือกthresholdด้วยlabel, testท้ายสุด; time-splitเมื่อเป็นsensorและไม่นำanomalytestfit scaler
16. zscore O(nd), IFขึ้นกับจำนวนต้นไม้และsubsample, LOFอาจO(n²d) หากbruteforce; รายงาน index,neighborcount,walltimeและpeakmemoryตาม n
17. X→trainpreprocess→rawscore→กำหนดทิศทางanomaly→validationthreshold→alert→PR/falsealertbudget; เก็บscoreต่อจุดเพื่อdebugไม่ใช่เฉพาะ0/1
18. zscoreเทียบ scipy.stats.zscore(ddof=0); IF/LOFไม่ควรคาด labelตรงเพราะเพียงใช้seedเดียวต่างimplementation ให้เทียบscore rankและqualityบนชุดเดียว
19. ตรึงscoresและlabels เปลี่ยนthresholdจากvalidationquantile.95→.99; จำนวนalertไม่ควรเพิ่ม แต่recallอาจลด; แสดงTP,FP,FNจริงทั้งสองค่า
20. บันทึกnormal-definition/score-direction/threshold/datawindow; monitor alertvolumeและdelayedlabels; ทดลองthresholdpersegmentลดfalsealertโดยrecallไม่ต่ำกว่าbaseline
## 17
1. trend คือระดับเฉลี่ยที่เปลี่ยนตามเวลา เช่น yₜ=2t+10 ไม่ใช่seasonalityเพราะไม่มีการวนรอบคงที่
2. seasonality คือpatternซ้ำตามคาบ เช่น [1,3,1,3,...] คาบ2; ต่างจากcycleเศรษฐกิจที่ระยะเวลาขึ้นลงไม่ตายตัว
3. weak stationarity ต้องmean,varคงที่และcovขึ้นกับlagไม่ขึ้นกับเวลา; randomwalkไม่stationaryแม้incrementstationary; ADFเป็นหลักฐานจำกัดไม่ใช่พิสูจน์
4. ACF lag k คือcorr(yₜ,yₜ₋ₖ); คาบ7มักมีpeakที่7,14; trendทำacfสูงยาวโดยไม่ได้แปลว่าARorderสูง
5. trendแยกด้วยระดับที่เคลื่อนไปเรื่อย ส่วนseasonalityลบด้วย seasonal difference yₜ−yₜ₋ₘ; [1,2,3,4] ordinarydifferenceได้[1,1,1]
6. positive lag1ทำ persistencebaselineสมเหตุผล แต่acfจากtrendปลอมได้; ตรวจdifferencedseriesและPACFก่อนเลือกlags
7. ARIMAเชิงเส้นอาจพลาดnonlinear/covariateและstructuralbreak; residualยังมีACFหรือforecastbiasเป็นสัญญาณตรวจ ไม่แก้ด้วยเพิ่มpอย่างเดียว
8. walk-forwardรักษาเวลาจริงแต่fitหลายครั้งแพง; expandingใช้ข้อมูลมากขึ้น,rollingปรับตัวต่อdriftได้แต่ทิ้งอดีต; horizon/gapต้องตรงการใช้งาน
9. `t=np.arange(len(y)); slope=np.sum((t-t.mean())*(y-y.mean()))/np.sum((t-t.mean())**2)`; y=[1,3,5] ได้slope2,intercept1
10. assert slope2สำหรับ[1,3,5],slope0สำหรับ[2,2,2],len<2ต้องreject; forecastt3ได้7 ห้ามrandomsplitลำดับเวลาเพื่อตรวจtrend
11. ใช้firstdifference `np.diff([1,3,6,10])` ได้[2,3,4]และrollingmean/varตรวจความคงที่; differencingเป็นการแปลงที่ช่วย ไม่ได้รับประกันstationarityทุกชุด
12. seed7 สร้างtrend+.5sin(2πt/7)+noise; walkforward horizon7 เปรียบเทียบlastvalue/seasonalnaive/ARIMAด้วยMAEและresidualACF; reportactualmetrics
13. featurelagต้องจับคู่ Xₜ=[yₜ₋₁,yₜ₋₂]กับtargetyₜ; shiftผิด1แถวแอบใช้currenttarget ให้เขียนtimestampsทุกcolumnและassertfeature_time<target_time
14. MAPEหารศูนย์เมื่อyจริง0; ใช้MAE/MASEหรือmaskที่ประกาศ; logtransformต้องy>0และinvertforecastพร้อมอธิบายbias
15. sorttime→trainpast→validationfuture→testlast; rollingfeaturesต้องshiftก่อนrolling; scalersfittrainและretrainตามข้อมูลที่มีจริงในแต่ละorigin
16. วัดfit/forecastต่อoriginแยกกัน; lagmatrixO(nL)memory,naiveforecastO(h); ARIMAcomplexityขึ้นกับsolver/orders จึงต้องrecordp,d,q,iterations
17. timestamps→resample/missingpolicy→lags/difference→fitpast→predictfuture→inverttransform→evaluatehorizon; อย่าใช้forecastสะสมเป็นobservedlagโดยไม่ระบุrecursiveforecast
18. scratchAR1OLSเทียบ statsmodels.AutoReg(lags=1,trend='c')ด้วยholdbackเดียวกันและtrainingwindowเดียว; forecastcoefficientsควรใกล้ภายในtolerance
19. คงsplit/horizon เปรียบเทียบARIMA(1,0,0)กับ(1,1,0); reportwalkforwardMAEและresidualACFไม่เลือกด้วยtrainAICอย่างเดียว
20. บันทึกtimezone/frequency/missingpolicy/origins/horizon/versions; monitorforecastbiasbyhorizon; ทดลองseasonalnaiveดีกว่าlastvalueในweeklyseriesโดยpairedabsoluteerrors
## 18
1. neuronรับx(d),w(d),b scalarคำนวณz=wᵀx+bแล้วa=φ(z); x=[1,2],w=[.5,−1],b=.5ให้z=−1,aReLU=0
2. linear layerทำneuronsหลายตัวพร้อมกัน: X(B,d)W(d,h)+b(h)→Z(B,h); B2,d3,h4ให้output(2,4)และparameters16
3. activationเป็นnonlinearfunctionเช่นReLUmax(0,z); ถ้าทุกชั้นlinearเท่านั้นW₁W₂รวมเป็นlinearเดียว ไม่เพิ่มnonlinearcapacity
4. forwardpassคือคำนวณactivationตามdependencyจากinputถึงlossโดยใช้weightsปัจจุบัน ไม่มีoptimizerupdateระหว่างอ่านผล
5. neuronเป็นหนึ่งoutput ส่วนlinearlayerรวมหลายneuronที่shareinput; weightorientationต้องประกาศ NumPy(d,h)ต่างจากtorchLinear(h,d)
6. เปลี่ยนwทำให้z,a,lossเปลี่ยนทันที; จะแก้weightsจากgradientต้องbackwardและstepแยกจากforwardให้ชัด
7. ตั้งweightsทุกhiddenunitเหมือนกันทำให้gradเหมือนกันและไม่breaksymmetry; ใช้Hevariance2/fan_inสำหรับReLUพร้อมตรวจactivationvariance
8. batchใหญ่ใช้matrixhardwareคุ้มแต่memoryเพิ่มและgradientnoiseลด; batch1SGDต่างจากfullbatchแม้จำนวนepochเท่ากัน ให้เทียบจำนวนupdatesด้วย
9. `z=np.dot([1,2],[.5,-1])+.5; a=max(0,z)` ได้−1,0; เปลี่ยนbเป็น2ได้z=.5,a=.5; เชื่อมผลก่อน/หลังที่neuronlab
10. assertz−1,a0และinputshapeเท่ากับw; bias2ให้a.5; numericfinitecheckและmismatchedwlengthต้องraiseแทนbroadcastผิด
11. `relu=np.maximum(z,0); sigmoid=np.where(z>=0,1/(1+np.exp(-np.abs(z))),np.exp(-np.abs(z))/(1+np.exp(-np.abs(z))))`; z=[−1,0,1]ให้ReLU[0,0,1]
12. seed7 สร้างXOR4จุด; trainMLP2→4→1และlinearmodelด้วยbatch4/2 คงinit/data; reportloss/accuracy/updatesจริงและactivationdeadfraction
13. X(B,d)@W(h,d)ผิดถ้าไม่transpose; target(B,)ลบprediction(B,1)จะกลาย(B,B); assertprediction.shape==target.shapeก่อนMSE
14. sigmoid/softmaxexpใหญ่overflow ให้stableformula/logitsloss; float32activationอาจoverflowก่อนlossจึงตรวจzของชั้นแรกที่nonfinite
15. splitก่อนnormalization; trainmodeใช้dropout,evalmodeปิดdropout; validationไม่updateweights; testครั้งท้ายหลังเลือกarchitecture
16. denseforward O(Bdh) และactivationO(Bh),weightsO(dh); backwardต้องเก็บintermediatesเพิ่ม; วัดB,d,hและpeakmemoryด้วยdtypeเดียว
17. X→Z₁=XW₁+b₁→A₁=ReLUZ₁→Z₂=A₁W₂+b₂→loss→backprop→update; ระบุshapesทุกลูกศรและแยกinferenceจากtraining
18. copyNumPyW.Tเข้าtorch.nn.Linear.weight,bเหมือนกัน,dtypefloat64 แล้วassertforwardallclose; ไม่เพียงseedเดียวเพราะinitalgorithmต่างกัน
19. เปลี่ยนเฉพาะzero→HeinitializationบนMLPเดิม; recordlossและunitoutputsหลังทุกupdate; symmetriczeroอาจเรียนbiasได้แต่hiddenunitsยังเหมือนกัน
20. บันทึกarchitecture/init/activation/lossreduction/seed/split; monitoractivation/gradnorm; ทดลองHeลดdeadReLUเทียบlargevarianceโดยวัดก่อนและหลังฝึกจริง
## 19
1. chainruleคำนวณdL/dx=(dL/du)(du/dx); u=2x,L=u²,x=3ได้u6,L36,dL/dx24
2. computationalgraphแยกoperationและdependency; x→u=x*x→L=u+x ที่x2ได้u4,L6และgradient2x+1=5
3. reversemodeเริ่มadjointloss1แล้วเดินย้อนgraph สะสมgradientทุกเส้นทาง เหมาะscalar lossกับparametersจำนวนมาก
4. gradientaccumulationคือบวกcontributionจากหลายเส้นทางหรือหลายmicrobatch; x*xมีcontributionxสองครั้งจึงได้2xไม่ใช่x
5. chainruleเป็นกฎอนุพันธ์ ส่วนgraphเป็นโครงสร้างที่ใช้จัดลำดับกฎ; graphอย่างเดียวไม่ให้gradientหากไม่กำหนดlocalderivative
6. ไม่zero_gradก่อนtrainingupdateทำให้gradientจากรอบเก่าค้าง; ตั้งใจaccumulatekbatchต้องscaleแต่ละloss/kก่อนstepเพื่อให้meanlossตรง
7. broadcastingbias(h)ไปBแถวมีbackwarddb=sum(dZ,axis=0); ไม่ใช่copydZ(B,h)กลับไปbias(h); axisผิดมักเงียบเมื่อB=h
8. finite differenceใช้2forwardต่อparameter ช้าO(P)และไวต่อepsilon; ใช้tinyfloat64smoothpointและเทียบrelativeerror ไม่ตรวจReLUkinkที่0
9. `x=3.; u=2*x; dLdu=2*u; dudx=2; dx=dLdu*dudx` ได้24; เทียบcentraldifferenceของ(2*x)**2ด้วยh1e−5
10. assertL36,dx24; testx−3ได้dx−24; fanoutL=x*x+xที่x2ได้5; ตรวจhหลายค่าที่ไม่เล็กจนroundoffครอบงำ
11. topo=[x,u,L]; grads[L]=1; reverseแต่ละnodeทำ `parent.grad += node.grad*local_derivative`; x*xต้องเก็บparentซ้ำหรือสะสมสองedgeอย่างถูกต้อง
12. ใช้MLP2→3→1float64seed7, zero gradsก่อนbackward, compareทุกweightกับfinite differences; accumulate2microbatchควรเท่ากับfullmeanbatchเมื่อscaleถูก
13. matmulbackwarddX=dZ@W.T,dW=X.T@dZ; รูปร่างdWต้องตรงW; targetbroadcastผิดทำlossgraphผิดแม้autogradคำนวณอนุพันธ์ของgraphนั้นถูก
14. finite differenceshเล็กเกินเกิดsubtractivecancellation; ลอง1e−4/1e−5/1e−6และใช้atol/rtol; sigmoidstableและfloat64สำหรับoracle
15. backwardเฉพาะtrainloss; validationno_gradและไม่แก้runningoptimizerstate; gradientchecksใช้syntheticdataไม่ต้องอ้างtestsetเพื่อเลือกhyperparameters
16. reversegraphO(numberofoperations)โดยประมาณเท่าforwardconstantfactorและเก็บactivation; finite diffO(P·forwardcost); วัดmemoryเมื่อretain_graphด้วย
17. forwardบันทึกinputs/outputs→lossseed1→reverse topological→sumadjoints→unbroadcast→parametergrads→optimizer; graphsharednodeต้องexecutebackwardหลังchildrenครบ
18. ใช้torchfloat64manualweightsและidenticalloss.mean; compareX/W/bgradsด้วยallclose; หากต่างเริ่มจากoutputgradก่อนย้อนทีละnode
19. เปลี่ยนbiasเป็นbroadcastแทนexplicittile; forwardและdbต้องตรงกัน db=sum(dZ); testB=3,h=2เพื่อจับsumaxisผิดที่B=hซ่อน
20. บันทึกgraph/reduction/dtype/h/tolerance; monitorgradnorm/NaN; ทดลองcheckpointingลดpeakmemoryเทียบbaselineโดยgradientต่างไม่เกินtoleranceและรายงานเวลาเพิ่ม
## 20
1. SGD θ←θ−ηg; L=θ²,θ2,g4,η.1ได้θ1.6,L2.56หลังหนึ่งupdate; เปิดหน้าเฉยๆต้องθ2,L4,updates0
2. momentumนิยามv←βv+g,θ←θ−ηv; v0=0,β.9 รอบแรกv4,θ1.6 รอบสองg3.2,v6.8,θ.92
3. RMSProp v←βv+(1−β)g²,θ←θ−ηg/(√v+ε); ลดstepของcoordinateที่gradientกำลังสองสะสมสูง ไม่ใช่เปลี่ยนloss
4. Adamใช้m,vและbiascorrection m/(1−β₁ᵗ),v/(1−β₂ᵗ); θ2,g4 รอบแรกη.1ให้θ≈1.9 ต่างจากSGD1.6
5. SGDขึ้นกับgradientปัจจุบัน ส่วนmomentumมีvelocityสะสม; คืนθเดิมแต่ไม่คืนvจึงไม่ใช่เริ่มการทดลองเดิม
6. Adamnormalizeด้วยRMSของgradientต่อcoordinateและfirstmoment; เปลี่ยนηแก้proposalของก้าวถัดไป ไม่ควรทำให้currentθเปลี่ยนจนกดstep
7. AdamWแยกdecay θ←(1−ηλ)θ−ηadam_update; การใส่λθในgradientก่อนadaptivepreconditionไม่เท่ากัน โดยปกติแยกbias/normจากdecay
8. scheduleเปลี่ยนηตามupdate/epoch; warmupลดearlyinstabilityแต่stepสั้นทำให้ช้า; ต้องระบุscheduler.stepก่อน/หลังoptimizerตามAPIไม่ให้off-by-one
9. `theta=2.; grad=2*theta; theta-=.1*grad` ได้1.6; lossก่อน4หลัง2.56; functionรับθ,gradient,ηแล้วreturnใหม่ไม่แก้historyเก่า
10. assertinitial2/4/0,step1เป็น1.6/2.56/1; η0ควรไม่เปลี่ยนθแต่completedstepเพิ่มตามนิยาม; ηติดลบrejectและbaselineต้องimmutable
11. RMSProp g4,β.9,v0=0 ได้v1.6,update .4/(sqrt1.6+ε)≈.316228,θ≈1.683772; testzero gradientและepsilonไม่เป็นศูนย์
12. seed7MLPเดียวกันเปรียบSGDmomentum/Adamโดยcopyinitialweights,samebatches; plotlossกับupdatesและηschedule อย่าreportคะแนนที่ยังไม่ได้run
13. momentbuffersต้องshape/dtype/deviceตรงparameterแต่ละตัว; โหลดcheckpointparameterใหม่แต่momentเก่าอาจอัปเดตผิดแม้shapeผ่าน ต้องเก็บstep/paramgroupด้วย
14. Adamεป้องกันหารศูนย์; FP16secondmomentunderflowได้ ให้optimizerstateFP32/ตรวจgradfiniteและloss scaling ไม่ใช้εใหญ่กลบbug
15. ใช้validationเลือกη/optimizer/decay; testไม่ใช้scheduleหรือearlystop; resumeต้องคืนweights,moments,step,RNGและdataloaderposition
16. SGDมีparameter-sizedupdateO(P),Adamเพิ่มm,vสองชุดO(P)memoryและelementwiseO(P)เวลา; วัดoptimizerstateจริงหลังfirststepเพราะlazyinit
17. currentθ→forwardloss→backwardg→momentupdate→biascorrection→θnew→completedupdates+1; initialθเป็นconfig ไม่ใช่currentθและห้ามretrainในrenderer
18. `torch.optim.SGD(lr=.1,momentum=.9)` gradient4แล้ว3.2ได้θ.92หลัง2stepsตรงนิยามv; Adamเทียบβ/ε/weightdecay/dtypeและstepเริ่ม1
19. เปรียบAdamL2กับAdamWโดยparameter2,gdata0,λ.1,η.1; AdamWdecayได้1.98 ส่วนAdamL2adaptiveupdate≈1.9 แสดงความไม่เท่ากันชัด
20. บันทึกinit/current/moments/step/schedule/seed; monitorgradnormและupdate-to-weight ratio; ทดลองwarmupลดNaNrunsโดยคงcompute budgetและreportหลายseed
## 21
1. ReLU(z)=max(0,z),derivative1เมื่อz>0,0เมื่อz<0; z0ไม่differentiableต้องประกาศsubgradientที่ใช้เช่น0
2. GELU(z)=zΦ(z); z1ได้≈.8413447,z−1≈−.1586553 เป็นsmoothgatingไม่ใช่ตัดnegativeทั้งหมด
3. SiLU(z)=zσ(z); derivativeσ(z)+zσ(z)(1−σ(z)); z0 output0 derivative.5 จึงไม่เหมือนReLUที่0
4. softmaxzᵢ=exp(zᵢ−maxz)/Σexp(zⱼ−maxz); logits[0,log3]ให้prob[.25,.75]และรวม1ตามclassaxis
5. ReLUnegativeเป็น0ส่วนGELUnegativeอาจไม่ศูนย์; GELUแพงกว่าแต่ไม่รับประกันaccuracyดีกว่า ต้องคุมarchitecture/init/data
6. softmaxเพิ่มlogitหนึ่งทำprobอื่นลดเพราะdenominatorร่วมกัน; บวกconstantทุกlogitprobคงเดิม ไม่ใช่independentsigmoid
7. cross-entropyคาดlogits(C)กับtargetclassint0..C−1 หรือdistributionตามAPI; softmaxก่อนCrossEntropyLossซ้ำทำgradientผิดนิยาม
8. contrastivelossต้องกำหนดpositive/negativeและtemperature; false negativesผลักข้อความความหมายเดียวออกจากกัน batchใหญ่เพิ่มnegativesและmemory
9. `np.maximum(np.array([-2.,0.,3.]),0)` ได้[0,0,3]; backwardใช้maskz>0กับupstreamgradientไม่overwritegradientจากbranchอื่น
10. assertoutputs[0,0,3],grads[0,0,1]ตามsubgradient0; inputemptyรักษาshapeและNaNต้องแจ้ง; finite differenceห่างจาก0เท่านั้น
11. `s=stable_sigmoid(z); out=z*s; grad=s+z*s*(1-s)`; z0ได้out0,grad.5; z1out≈.731059
12. seed7XORเปรียบGELU/ReLUคงweights/batches/lr; softmaxCEกับcontrastiveเป็นคนละtaskต้องกำหนดlabels/pairsก่อน อย่าลากlossเปรียบscaleตรงๆ
13. logits(B,C),labels(B,)long; softmaxaxis−1 ไม่ใช่batchaxis0; probabilityแต่ละsampleรวม1 และpaddingtokensต้องignoreในloss
14. logsumexp=maxz+logΣexp(z−maxz); CE=logsumexp−zy; logits[1000,1001]ไม่ควรNaN; log(softmax)ตรงๆอาจlog0
15. tuneactivation/temperatureบนvalidation; classweightsคำนวณtrainเท่านั้น; testใช้calibrationที่freezeแล้วไม่เลือกthresholdหลังเห็นtest
16. activationsO(numberofelements); fullcontrastivesimilarity(B,B)ใช้O(B²d)เวลาO(B²)memory; วัดbatchsizeและmasksemanticsให้ตรง
17. logits→validmask→stable logsumexp→selectedclasslogit→lossmeanvalidtokens→gradient(p−onehot)/N; อย่าmeanรวมpadding
18. torch.nn.functional.cross_entropyเทียบNumPylogsumexpด้วยfloat64; GELUexactเทียบapproximate='none'ไม่ใช้tanhapproxโดยไม่บอก
19. คงlogits/labels เปลี่ยนlossreductionmean→sum; gradientต้องเพิ่มfactorN; ผลต่างจากηเดิมไม่ใช่หลักฐานว่าsumlossเรียนรู้ดีกว่า
20. บันทึกactivationapprox/lossaxis/mask/reduction/temperature; monitorentropy/deadunits; ทดลองSiLUเพิ่มconvergenceในMLPเล็กภายใต้updatesเท่ากันพร้อม3seeds
## 22
1. torch.Tensorเก็บค่าพร้อมshape,dtype,device; `torch.tensor([[1.,2.],[3.,4.]])` shape(2,2),float32โดยdefaultตามconfig ไม่ได้requires_gradอัตโนมัติ
2. autogradสร้างgraphของoperationบนtensorrequires_grad; x2,L=x²,backwardให้x.grad4; graphกับgradbufferมีlifecycleต่างกัน
3. nn.Moduleลงทะเบียนparameters/submodulesเพื่อให้parameters(),state_dict(),to(),train()/eval()ทำงาน; listPythonธรรมดาไม่แทนModuleList
4. DataLoaderแปลงDatasetเป็นbatchesและcollate/shuffle; X5แถว,batch2ให้3batchesหากdrop_last=False seedของworkerต้องตั้งแยก
5. tensorคือข้อมูลและoperation ส่วนautogradคือระบบคำนวณอนุพันธ์; integerclasslabelsไม่requires_gradแต่ยังใช้ในlossได้
6. shuffleเปลี่ยนลำดับSGDไม่เปลี่ยนsetข้อมูล; drop_last=Trueตัดbatchท้ายจึงเปลี่ยนจำนวนexamples/updates; validationควรครอบคลุมทุกsample
7. ลืมoptimizer.zero_gradทำgradสะสมโดยไม่ตั้งใจ; ลืมstepไม่มีweightchange; eval()ไม่เท่ากับno_grad ต้องใช้ทั้งสองตามบทบาท
8. state_dictเก็บparameter/buffersไม่เก็บarchitecturecode; checkpointresumeควรมีoptimizer/scheduler/RNG/epoch; โหลดเฉพาะแหล่งเชื่อถือและตรวจmetadata
9. `x=torch.tensor([1.,2.],requires_grad=True); loss=(x*x).sum(); loss.backward()` ได้loss5,x.grad[2,4]บนCPU
10. assertloss5,grad[2,4],shape(2,),devicecpu; zero_gradแล้วbackwardรอบใหม่ต้องไม่เป็น[4,8]; ชุดทดสอบต้องrunจริงไม่skipเพราะtorchหาย
11. สร้าง nn.Linear(2,1,bias=True),copyweight[[1,2]],bias[.5]; input[[3,4]]ให้11.5; ตรวจnamed_parametersมีweight,bias
12. manual_seed7,tinyregressionDataset; ฝึกtrainloader,batch2,savecheckpoint แล้วloadmodelใหม่eval predictionsต้องallcloseก่อนและหลังload
13. Linearคาดlastdim=in_features; output(B,1)กับtarget(B,)broadcastผิด ให้target.unsqueeze(1); CrossEntropytargetต้องlong(B,),ไม่float(B,1)
14. ใช้BCEWithLogitsLossแทนsigmoidแล้วBCELossสำหรับextreme logits; gradclipหลังbackwardก่อนstep และตรวจnonfiniteก่อนoptimizerstateเสีย
15. model.train→zero_grad→forward→loss→backward→step; validationmodel.eval+torch.no_grad; เลือกbestcheckpointด้วยvalแล้วเปิดtestครั้งท้าย
16. CPUใช้perf_counterหลังwarmup; CUDAต้องsynchronizeก่อน/หลังจับเวลา; ระบุbatch,dtype,deviceและpeakallocatedไม่เอาcachememoryตีเป็นmodelsize
17. Dataset→DataLoader→tensor.to(device)→Module.forward→loss→autograd→optimizer→checkpoint; state_dictต้องมีpersistentbuffersเช่นBatchNormrunningmean
18. copyNumPyW.T,bเข้าLinearfloat64แล้วเทียบforwardและgradientMSEmean; frameworktestผ่านCPUเป็นหลักฐานruntimeไม่ใช่ตรวจโค้ดผู้เรียน
19. เปลี่ยนเฉพาะzero_grad(set_to_none=True/False)และตรวจupdatesเท่ากันเมื่อทุกparamมีgradient; unusedparameterอาจมีsemanticsต่างต้องเพิ่มtest
20. บันทึกtorch/python/device/seed/RNG/split/commit; ทดลองnum_workers0→2เพิ่มthroughputหรือไม่โดยprediction/evaluationตรงและรายงานstartupcost
## 23
1. tf.Tensorเก็บndarrayimmutableที่มีshape,dtype,device; parameterที่แก้ค่าใช้tf.Variable; [[1,2],[3,4]]float32shape(2,2)
2. GradientTapeบันทึกoperationsในcontextเพื่อตรวจgradient; x=tf.Variable(2.),loss=x*x ให้tape.gradient(loss,x)=4 นอกtapeไม่ถูกบันทึก
3. KerasModelรวมlayersและtrainablevariables,callนิยามforward; Dense(1)รับlastfeaturedimและต้องbuild/callก่อนเข้าถึงweights
4. tf.data.Datasetจัดmap/shuffle/batch/prefetch; cacheก่อนrandomaugmentationต่างจากcacheหลังaugmentationเพราะหลังcacheจะตรึงผลสุ่ม
5. Tensorคือค่าที่ส่งในgraph ส่วนGradientTapeเก็บความสัมพันธ์เพื่ออนุพันธ์; watchtensorconstantเองหากต้องการgradientไม่ใช่Variable
6. shufflebufferเล็กสลับไม่ทั่วdataset; repeatก่อนbatchอาจข้ามepochboundary ต้องกำหนดsteps_per_epochและไม่repeatvalidationไม่สิ้นสุด
7. EarlyStoppingmonitorval_lossและrestore_best_weightsตามเจตนา; ไม่ใช้testlossเลือกcheckpoint; callbackผิดชื่อmetricจะไม่หยุดตามที่คาด
8. checkpointweightsช่วยreproduceinferenceแต่resumeoptimizerต้องบันทึกstateด้วย; ตรวจroundtripoutputและstepอย่าใช้เพียงfileexists
9. `x=tf.Variable([1.,2.]); with tf.GradientTape() as t: loss=tf.reduce_sum(x*x)` แล้วgradientได้[2,4],loss5; indentationในPythonต้องถูก
10. assertloss5,gradient[2,4],shape(2,),finite; gradientของconstantนอกwatchควรNoneไม่ตีเป็น0; CPUtestต้องไม่skipcoreTensorFlow
11. Dense1weights[[1],[2]],bias[.5],input[[3,4]]ให้11.5; set_weightsหลังbuildและtraining=Falseสำหรับevaluation
12. tf.keras.utils.set_random_seed(7); tf.data batch2tinyregression; train2epochs,save/load.kerasแล้วpredictionallcloseและตรวจoptimizeriterationsเมื่อresume
13. KerasDensekernel(d,h)ต่างจากPyTorch(h,d); target(B,)กับoutput(B,1)ต้องreshape; variablebatchdimNoneไม่เท่ากับbatchจริง0
14. SparseCategoricalCrossentropy(from_logits=True)รับrawlogits; from_logitsFalseกับlogitsทำlossผิด ให้ตรวจfinalactivation/lossร่วมกัน
15. fittraindatasetพร้อมvalidation_dataแยก; preprocessadaptเฉพาะtrain; callbackmonitorvalidationและevaluate testหลังเลือกmodelเสร็จ
16. warmuptracedfunctionก่อนจับเวลา; `.numpy()`บังคับmaterialization; แยกtracecompiletimeจากsteady-stateและระบุCPU/threadconfig
17. tf.data→Model.call→loss→GradientTape.gradient→optimizer.apply_gradients→metrics→callbacks→checkpoint; Nonegradต้องหาว่าoperationหลุดtapeตรงไหน
18. ตั้งDenseweightsจากNumPyแบบ(d,h),dtypefloat64,lossreductionเดียวกัน; assertforward,dW,dbทั้งKeras/Torch/NumPyตรงบนtinyinput
19. เปลี่ยนEarlyStoppingpatience0→2โดยคงweights,epochorder,seed; รายงานbestepochกับstop_epochแยกและbestvalscoreไม่อ้างทุกrunดีขึ้น
20. บันทึกTF/Keras/python/seed/datasetpipeline/checkpointconfig; ทดลองprefetchเพิ่มexamples/secโดยfiniteepochและevaluationเท่าเดิม CPUผ่านไม่ยืนยันGPUintegration
