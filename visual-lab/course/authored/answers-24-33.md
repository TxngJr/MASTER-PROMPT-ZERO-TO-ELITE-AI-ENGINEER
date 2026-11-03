# เฉลยเฉพาะข้อ: neural mechanisms
## 24
1. convolutionในDLมักเป็นcross-correlation: Yᵢⱼ=ΣₐᵦXᵢₛ₊ₐ₋ₚ,ⱼₛ₊ᵦ₋ₚKₐᵦ+b; filterเลื่อนใช้weightsเดียวกันทุกตำแหน่ง ไม่ใช่matrixmultiplyทั้งภาพครั้งเดียว
2. kernelคือweights(Kh,Kw,Cin,Cout); kernel2×2[[1,0],[0,−1]]กับpatch[[1,2],[3,4]]ให้1−4=−3ก่อนbias
3. strideคือระยะเลื่อนpatch; H4,K2,p0,s2ให้Hout=floor((4−2)/2)+1=2 จึงมี4ตำแหน่งในoutput2×2
4. paddingเติมขอบก่อนเลือกpatch; zero-padding1เปลี่ยนH4,K2,s1จากHout3เป็น5 และค่าขอบผสมศูนย์ไม่เท่ากับจุดในภาพ
5. convolutionเป็นoperationส่วนkernelเป็นparameter; แก้kernelweightหนึ่งตัวเปลี่ยนทุกoutputpatchที่เห็นตำแหน่งสัมพัทธ์นั้น
6. paddingเพิ่มโอกาสให้ขอบมีส่วนร่วมและขนาดoutput แต่เติมข้อมูลสมมติ ไม่ควรอ้างว่าเพิ่มinformationจริง; inspectpatchที่cornerกับcenter
7. maxpoolingทิ้งตำแหน่งย่อยและdetail; patternเล็กอาจหาย ยก[1,0;0,0]กับ[0,1;0,0]ให้pooled1เหมือนกัน
8. receptivefieldใหญ่เห็นบริบทมากขึ้นแต่ละเอียดเชิงพื้นที่ลดเมื่อstrideสูง; คำนวณrnew=rold+(k−1)jump,jumpnew=jump·s
9. X=np.arange(1,10).reshape(3,3),K=[[1,0],[0,−1]]; loop4patchesแล้วsum(patch*K)ได้[[-4,-4],[-4,-4]]
10. assertoutput[[-4,-4],[-4,-4]],shape(2,2),bias.5เพิ่มทุกช่อง.5; invalidindex8สำหรับoutput2×2ต้องraiseและsnapshotเดิมไม่เปลี่ยน
11. looprowcolใช้start=i*stride-paddingก่อนดึงpatch; outputindex3ของ2×2คือ(row1,col1) ไม่ใช้index%4เพื่อรับ8อย่างเงียบๆ
12. seed7image4×4kernel2×2; เปรียบpadding0/1และstride1/2,recordactualoutput,receptivefield; จำนวนpatchต้องตรงสูตรทุกconfiguration
13. PyTorchimage(B,C,H,W),TensorFlowdefault(B,H,W,C); Cinของkernelต้องตรงinput ไม่ใช่นับคอลัมน์ตารางว่าเป็นtensorchannels
14. productสะสมในfloat16อาจoverflow/roundoff; ใช้float64tinyoracleแล้วfloat32tolerance; inputNaNต้องรายงานตำแหน่งไม่วาดกราฟต่อเป็น0
15. splitตามpatient/sourceก่อนaugmentation; fitnormalizationtrain; validation/testไม่มีrandomcropที่เปลี่ยนlabelsemanticโดยไม่ประกาศ
16. directconv O(BHoutWoutCoutCinKhKw),im2colเพิ่มmemory; วัดkernel/stride/dtypeพร้อมactivationและparameterbytesไม่เฉพาะweights
17. image→pad→patch→elementwiseproduct→sumCin/Kh/Kw→bias→outputcell→activation; selectedcellเน้นpatchเดียวกันในtensorและตาราง
18. torch.conv2dใช้crosscorrelationไม่flipkernel; copyKเป็น(1,1,2,2),Xเป็น(1,1,3,3)แล้วallcloseกับข้อ9
19. คงCNN/split/trainingbudget เปลี่ยนmaxpool→avgpool; reportaccuracyและlatencyจริงพร้อมfeaturemapของtinyinput ไม่สรุปaccuracyดีขึ้นเสมอ
20. บันทึกinputlayout/kernel/stride/padding/dilation/seed; ทดลองลดstrideในsmallobjecttaskเพิ่มrecallภายใต้memorybudget พร้อมbaselineและ3seeds
## 25
1. hiddenstate hₜเป็นเวกเตอร์สรุปอดีตที่อัปเดตตามเวลา; h₀0,hₜ=tanh(.5hₜ₋₁+xₜ) กับx₁1ได้h₁≈.761594
2. recurrenceคือใช้function/weightsเดียวกันทุกt: hₜ=φ(xₜWx+hₜ₋₁Wh+b); ต่างจากlayerแต่ละเวลาที่มีweightsคนละชุด
3. BPTTคลี่recurrenceตามเวลาแล้วbackpropสะสมgradientของsharedweightsทุกt; truncatedBPTTตัดdependencyย้อนเกินwindow
4. vanishinggradientเกิดproductJacobianหลายเวลา norm<1ทำให้สัญญาณอดีตเล็กลง เช่น .5^20≈9.54e−7 ไม่ใช่เพียงlossต่ำ
5. hiddenstateเป็นค่าชั่วคราว(h)ส่วนrecurrenceเป็นกฎที่สร้างค่าถัดไป; stateจากsequenceหนึ่งไม่ควรไหลเข้าอีกsequenceโดยไม่ตั้งใจ
6. gradientหายทำlong-rangelearningยากแม้forwardยังเห็นทุกtoken; ตรวจgradnormตามเวลา,ใช้gatedRNN/shorterpathsโดยไม่รับประกันหายทั้งหมด
7. paddingต้องmasklossและhiddenupdateตามlength; batchที่shuffleแต่carryhเดิมจะผูกคนละsequenceเข้าด้วยกัน ต้องreset/detachตามboundary
8. teacherforcingใช้ground-truthpreviousoutputฝึกเร็วกว่าป้อนgeneratedtoken แต่inferenceเจอerrorสะสม/exposurebias; ประเมินfreerunningด้วย
9. `h=0.; states=[]; for x in [1.,0.]: h=np.tanh(x+.5*h); states.append(h)` ได้[.761594,.363400]โดยประมาณ
10. assertstateแรกtanh1,stateสองtanh(.5*tanh1),emptysequenceคืนh0ตามcontract; sequencebatchresetต้องให้ผลเหมือนรันแยก
11. recurrencehₜ=a hₜ₋₁+xₜ,L=.5h₂²,h₀0,x=[1,2],a=.5: h1=1,h2=2.5,dL/da=2.5; reverseสะสมlocalcontribution
12. seed7sequencecopytask เปรียบshort/longwindowคงRNNweightsแล้วrecordgradnormfirsttoken,teacherforcedlossและfreerunaccuracy ไม่เอาlossอย่างเดียว
13. PyTorchRNNdefault(T,B,D)ต่างbatch_first(B,T,D); hidden(layers,B,H)ไม่ใช่(B,T,H); testT≠Bจับaxisสลับ
14. recurrentJacobianใหญ่ทำexplodinggradient; clipglobalnormหลังbackwardและตรวจnonfinite; clipไม่แก้graph/labelbugหรือvanishinggradient
15. splitตามsequence/timeก่อนwindowingเพื่อไม่overlaptrain/test; scalingfittrain; validationfreerunninghorizonเท่าการใช้งาน
16. denseRNN O(TB(DH+H²)),backpropactivationO(TBH); เวลาtserialแม้batchparallel จึงวัดTและbatchต่างกัน
17. tokens→embedding→h0→recurrenceh1..hT→head→maskedloss→BPTT→updatesharedW; แยกteacherforcedinputจากinferencegeneratedinput
18. copyweights/biasesเข้าtorch.nn.RNN(tanh,batch_first=True),zeroextra biasให้ตรง; comparestatesทุกtและgrad ไม่แค่finaloutput
19. เปลี่ยนpaddingbatch→packedsequenceคงlengths/weights; validtimestepsต้องตรงและmeasuretokens/secโดยไม่นับpaddingเป็นprocessedusefultoken
20. บันทึกsequenceboundary/hreset/detach/lengths/horizon; ทดลองtruncationwindow10→30ช่วยlongdelayaccuracyแต่memoryเพิ่มเท่าไรพร้อมseedหลายค่า
## 26
1. forgetgate fₜ=σ(Wfx+Ufhprev+bf)ควบคุมส่วนcprevที่เก็บ; cprev2,f.8ให้ส่วนเดิม1.6 ก่อนรับcandidateใหม่
2. inputgate iₜควบคุมcandidategₜที่เขียนลงcell; i.25,g.4ให้ส่วนใหม่.1 จึงcnew1.7เมื่อส่วนเก่า1.6
3. outputgate oₜควบคุมhidden hₜ=oₜtanh(cₜ); c1.7,o.5ให้h≈.467705 ไม่แก้cellstateโดยตรง
4. cellstate cₜ=fₜcprev+iₜgₜมีadditivepathที่ช่วยgradientไหล; ไม่ได้รับประกันไม่มีvanishingเพราะproductfยังเล็กได้
5. forgetgateลด/เก็บอดีต ส่วนinputgateเลือกข้อมูลใหม่; f1,i0รักษาcellเดิม ขณะที่f0,i1แทนด้วยcandidate
6. เปลี่ยนcellstateส่งผลhผ่านtanhและoutputgate; cใหญ่มากtanhอิ่มตัวจึงhแทบไม่เปลี่ยนแม้cเปลี่ยนมาก
7. GRUupdategateมีสองconvention z·hprev+(1−z)gหรือกลับกัน ต้องอ่านAPIก่อนcopyweights; ค่าz1อาจหมายkeepหรือreplaceต่างกัน
8. resetgateลดอดีตในcandidate ช่วยเริ่มpatternใหม่แต่ไม่ใช่resetactualstateอัตโนมัติ; GRUไม่มีcellstateแยกและparametersน้อยกว่าLSTMทั่วไป
9. `sigmoid=lambda z:1/(1+np.exp(-z)); f=sigmoid(np.log(4.)); retained=f*2` ได้f.8,retained1.6
10. assertf.8,retained1.6; logits0ให้gate.5; teststable±1000 finite; shapegateตรงcprevและbatchไม่ปนกัน
11. `c=.8*2+.25*.4; h=.5*np.tanh(c)` ได้c1.7,h≈.467705; outputgate0ให้h0แต่cยัง1.7
12. seed7copysequence เปรียบLSTM/GRUhiddenmatchedparameterbudget; loggates/cellnorm/validationaccuracyและแยกการเปลี่ยนresetbiasทีละตัว
13. gateconcatorderเช่นPyTorchLSTM i,f,g,o ต้องsplitlastdim4H; h,cเป็น(layers,B,H); transposeผิดทำhiddenunitsสลับโดยไม่จำเป็นต้องerror
14. sigmoidใช้stablelogits,cellstateสะสมโตได้; checkcและhแยกและclipgrad; อย่าclipcellโดยเงียบเพราะเปลี่ยนกลไก
15. แยกsequenceก่อนสร้างwindows,reseth/cตามboundary; teacherforcingonlytrainและvalidationfreerun; checkpointต้องเก็บarchitecturegateconvention
16. LSTMประมาณ4denseprojectionsO(TBH(D+H)),GRU3; activationgatesเพิ่มmemoryO(TBH); reportparametercountและwalltimeจริง
17. concat[x,hprev]→4projections→σf/σi/tanhg/σo→c=f*cprev+i*g→h=o*tanhc; backwardadditivecellpathต้องรวมจากfuturecและh
18. torch.nn.LSTMCellcopygateweightsตามi,f,g,o,splitbiasให้ผลรวมตรง; compareh,cหนึ่งก้าวและหลายก้าวfloat64ก่อนtraining
19. คงGRUทุกอย่างเปลี่ยนupdatebias−2→2โดยconventionเดิม; วัดhretentionหลังinputzeroและtaskmetric ไม่สรุปgateสูงเรียนดีเสมอ
20. บันทึกgateorder/biasinit/statepolicy/seed; ทดลองforgetbias1เทียบ0ช่วยdelay20taskที่updatesเท่ากันและreportgate saturation
## 27
1. encoder fφแปลงx(d)เป็นlatentz(k); linearW=[[1],[1]]/2,x=[2,4]ให้z3; VAEencoderให้μและlogvarianceแทนzเดียว
2. latentspaceคือพื้นที่representationที่decoderอ่าน; kน้อยบังคับcompressionแต่ยังอาจจำข้อมูลเมื่อmodelcapacityสูง
3. decoder gθแปลงzกลับxhat; z3,W=[1,1],b[0,0]ให้[3,3] ไม่ใช่inverseสมบูรณ์ของทุกencoder
4. reconstructionMSEmean((xhat−x)²); x[2,4],xhat[3,3]ได้1; BCEเหมาะbinary/normalizedlikelihoodตามนิยามไม่ใช่ใช้กับทุกข้อมูล
5. encoderเป็นmappingที่มีparameters ส่วนlatentเป็นผลลัพธ์ของmapping; เปลี่ยนlatentdimensionเปลี่ยนทั้งencoderoutputและdecoderinput
6. lossreductionเปลี่ยนscalegradient; reconstructionดีอาจเก็บnoiseและไม่ให้latentที่sampleได้ จึงต้องประเมินgenerationด้วย
7. KLสูงเกินทำposteriorcollapse μ≈0,σ≈1และdecoderไม่ใช้z; KLต่ำเกินทำlatentholes ให้ดูKLต่อdimension/activeunits
8. reparameterization z=μ+exp(.5logvar)ε,ε~N(0,I)ทำgradientผ่านμ/σได้ แต่samplevarianceยังมีและไม่ใช้กับdiscretesampleตรงๆ
9. `x=np.array([2.,4.]); z=x@np.array([.5,.5])` ได้3; VAEให้μ3,logvar0,ε−1ได้z2ด้วยfixedε
10. assertz3,decoder[3,3],MSE1; fixedεทำreparameterizationdeterministic; KLของμ0/logvar0ต้อง0และshapeμ==logvar
11. `xhat=np.array([z,z]); loss=np.mean((xhat-[2,4])**2)`; z3loss1, z2loss2; คำนวณgradientdL/dz=2mean(xhat−x)=0ที่3
12. seed7toy2D เปรียบAEกับVAEβ0/.1/1คงcapacity; recordreconstruction/KL/seedsและsamplequalityไม่เทียบtotallossคนละนิยามตรงๆ
13. encoderlogvarต้องshape(B,k)เท่าμ ไม่broadcast(B,1)โดยไม่ตั้งใจ; decoderoutputshapeตรงinputและMSEreduceaxisตามsample/feature
14. exp(logvar)overflowได้; checkencoderoutput/logvarและlearningrateก่อนclampพร้อมประกาศbounds; KLสูตร−.5Σ(1+logvar−μ²−exp(logvar))
15. splitก่อนnormalization,validationreconstructionใช้fixedεหรือหลายsamplesรายงานvariance; testgenerationsไม่เลือกเฉพาะตัวอย่างสวย
16. denseAE O(Bdk)ต่อlayer,latentO(Bk); VAEเพิ่มmean/varianceheadsและsampling; reportactivation/optimizerstateไม่เอาlatentbytesแทนtrainingmemory
17. x→encoderμ/logvar→ε→z→decoderxhat→reconstruction+βKL→backpropผ่านreparameterization→update; evaluationgenerationเริ่มzpriorไม่ใช่encodex
18. compareNumPyKLกับtorch.distributions.kl_divergence(Normal(μ,σ),Normal(0,1)).sum(-1)โดยreductionเดียวกัน,fixedεforwardallclose
19. เปลี่ยนβ0→1โดยseed/capacity/updatesคงที่; reportreconstructionและKLแยกกับactiveunits ผลไม่จำเป็นmonotonicในstochastictraining
20. บันทึกlatentdim/prior/reconlikelihood/reduction/β/seed; ทดลองKLwarmupลดposteriorcollapseโดยactiveunitsเพิ่มและreconstructionไม่เกินbudget
## 28
1. generator G(z)แปลงnoiseเป็นsample; tinyG(z)=2z+1,z[−1,0,1]ได้[−1,1,3]; outputdistributionขึ้นกับpriorและweights
2. discriminator D(x)ให้probหรือlogitว่าsampleจริง; sigmoidlogit0ได้.5 ไม่ใช่qualityscoreที่compareข้ามtrainingrunsได้ตรงๆ
3. minimax minGmaxD ElogD(real)+Elog(1−D(G(z))); Dmaximizeแต่implementationมักminnegative และGminsecondtermอาจgradientเล็กเมื่อDมั่นใจfake
4. non-saturatingLG=−ElogD(G(z)); Dfake.1ได้2.302585แทนlog(.9)=−.105361ของminimaxG ช่วยgradientเมื่อDfakeต่ำ
5. Gสร้างfakeและรับgradientผ่านDไปG ส่วนDจำแนกreal/fake; updateDต้องdetachfake,updateGต้องไม่detachfakeแต่freezeDparametersได้
6. Dแรงเกินอาจให้Ggradientใช้ไม่ได้,อ่อนเกินfeedbackไม่มีความหมาย; ดูsamples/diversityและgradnormร่วมกับสองloss ไม่ตีความloss.69ว่าดีแน่
7. modecollapseคือGสร้างได้ไม่กี่modeแม้realมีหลายกลุ่ม เช่นnoiseทุกตัวออก0; lossGอาจดีแต่pairwisediversityต่ำ
8. capacityมากเพิ่มความสามารถแต่Doverfitง่าย; regularization/augmentation/learningrateต้องคุม และไม่ใช้trainingDaccuracyเป็นsamplequalitymetric
9. `z=np.array([-1.,0.,1.]); fake=2*z+1` ได้[−1,1,3]; ตรวจnoisevectorต่างให้outputต่างก่อนอ้างว่าGเรียนdistribution
10. assertfake[-1,1,3],shape(3,),fixedseedrepeatable; setweight0ให้constant1เพื่อเป็นcollapsecounterexample ไม่อ้างtestนี้พิสูจน์trainingสำเร็จ
11. Dreal.8,Dfake.2: LD=−log.8−log.8=.446287ตามsumสองexpectations,LG=−log.2=1.609438; ถ้าmeanรวมทั้งสองbatchLDจะหาร2ต้องระบุ
12. seed7realmixture1D±2; alternateD/G100stepsและlogLD/LGพร้อมfakehistogramทุก20steps; reportmodecoverageแม้trainingไม่เสถียร
13. Dlogits(B,1)กับlabels(B,)broadcast(B,B)ผิด; ones_like(logits)ทำlabelsตรงshape; Gnoise(B,zdim)ไม่ใช่จำนวนsamplesเท่ากับzdim
14. ใช้BCEWithLogitsLoss/softplus: LD=softplus(−real_logits)+softplus(fake_logits),LG=softplus(−fake_logits); หลีกเลี่ยงlog(sigmoid)ที่อาจlog0
15. trainGANด้วยrealtrain; validationเลือกcheckpointด้วยmetricที่กำหนดก่อน; testqualityรายงานunselectedsamplesหลายseedและไม่trainingonheldout
16. วัดDstepและGstepแยก,warmupและdevice sync; ต้องเก็บDgraphสำหรับGbackwardแต่ไม่optimizerD; peakmemoryขึ้นกับfakebatchและactivation
17. samplez→G→fake.detach→Dloss→Dstep; samplez→G→Dwithoutdetach→Gloss→Gstep; zerogradของแต่ละoptimizerก่อนbackward
18. logitsreal=log4,fake=−log4เทียบsoftpluslossNumPyกับtorchBCEWithLogitsLosssummeans; assertgradientthroughGมีและDstepไม่เปลี่ยนGweights
19. เปลี่ยนเฉพาะDlearningrateหรือgradientpenaltyหนึ่งอย่าง; คงnoise/seed/budgetและวัดmodecoverage/qualityหลายruns ไม่เลือกดีที่สุดrunเดียว
20. บันทึกprior/architectures/lossconventions/update-ratio/seed; ทดลองspectralnormลดcollapsefrequencyบนmixturetoyโดยquality/diversityและwalltimebudgetเดียว
## 29
1. query Q=XWQแทนสิ่งที่tokenตำแหน่งนั้นกำลังค้น; X(3,2),WQ(2,2)ได้Q(3,2),ไม่ใช่vectorคำถามจากผู้ใช้เสมอ
2. key K=XWKเป็นaddressที่ใช้จับคู่query; QKᵀให้score(queries,keys),ไม่เอาKเป็นweightedcontentก่อนsoftmax
3. value V=XWVเป็นcontentที่นำมาถัวเฉลี่ยด้วยA; เปลี่ยนWVอย่างเดียวAคงเดิมแต่outputAVเปลี่ยน
4. scaledattention A=softmax(QKᵀ/√dk+mask),O=AV; scaleลดvarianceของdotproductเมื่อdimensionสูง ไม่ใช่normalizeV
5. queryถาม/keyจับคู่; crossattentionQจากdecoder,KจากencoderจึงTqกับTkต่างได้ ส่วนselfattentionมาจากsequenceเดียวแต่projectionsคนละตัว
6. Q=[1,0],K=[[1,0],[0,1]],dk2ให้scores[.707107,0],weights≈[.669762,.330238]; weightsรวม1ทุกqueryที่ไม่fullymasked
7. causalmaskต้องปิดfuturej>iก่อนsoftmax; maskหลังsoftmaxแล้วไม่renormalizeทำrowmass<1; inferencecachepositionต้องตรงabsoluteoffset
8. multiheadแบ่งfeaturesให้headsแต่ละชุดเรียนrepresentationต่างกัน; dmodelต้องหารheadsลงตัวและcostscoreO(BhT²); ไม่คัดลอกweightsheadเดียวแล้วอ้างหลายกลไก
9. X=[[1,0],[0,1],[1,1]],WQ=Iได้Q=X; เปลี่ยนWQ00เป็น2ได้Q[[2,0],[0,1],[2,1]]ก่อนคำนวณscoresใหม่
10. assertQshape(3,2),changingWQchangesQ/scores,changingWVdoesnotchangeA; causalfuturemass0และallrowsum1; fullymaskedrowต้องerror
11. คำนวณV=XWVแล้วO=A@V; A=[.25,.75],V=[[2,0],[0,4]]ให้O=[.5,3] ไม่ใช่เลือกvalueของmaxweightอย่างเดียว
12. seed7X/projectionsfixed; compareheads1/2โดยdk2/1,recordQ/K/V/A/O; เปลี่ยนheadsอย่างเดียวแบ่งfeatureใหม่จึงoutputอาจเปลี่ยนตามจริง
13. Q(B,h,Tq,dk)@K.transpose(−2,−1)→(B,h,Tq,Tk); maskต้องbroadcastตรงaxes ไม่สลับqueryกับkey; Vlastdimdvอาจต่างdk
14. subtractrowmaxก่อนexp; fullymaskedrowmax−∞ทำNaNจึงrejectหรือกำหนดemptyrowpolicy; logitsfp16อาจoverflowก่อนsoftmaxให้accumulateFP32
15. fitprojectionsบนtrain; causallanguagevalidationห้ามfutureattention; paddingmaskและlossmaskคนละหน้าที่ต้องตรวจทั้งคู่
16. fullattention O(BhT²dk)เวลาและO(BhT²)scorememory; projectionsO(BTdmodel²); fusedattentionลดmaterializedmemoryแต่ไม่เปลี่ยนสมการพื้นฐาน
17. X→QKVprojections→reshape/splitheads→scaledQKᵀ→mask→rowsoftmax→AV→concat→WO; ใช้multiheadlabดูtensorทุกขั้น
18. torch.scaled_dot_product_attentionรับbooleanmasksemanticsต่างบางAPI; convertmaskชัดเจน,dropout_p0,dtypefloat64แล้วcompareQKV/O
19. คงQKVเปลี่ยนmasknone→causal; outputtoken0ต้องอ่านเฉพาะV0และfuturemass0,tokenท้ายอาจเท่าเดิม; gradientห้ามไหลไปfutureinput
20. บันทึกprojectionweights/headsplit/scalefactor/mask/dtype; ทดลอง2headsช่วยcopy-two-relationsเทียบ1headด้วยparameterbudgetใกล้กันและholdouttask
## 30
1. selfattentionให้แต่ละtokenอ่านtokensในsequenceเดียวผ่านQKVที่เรียนได้; permutationequivariantหากไม่มีpositionและmask
2. residual y=x+F(x)ให้เส้นทางidentityสำหรับข้อมูล/gradient; shapesต้องตรงกันหรือมีprojectionที่ระบุ ไม่ใช่concat
3. LayerNormnormalizefeaturesต่อtoken: (x−μ)/sqrt(var+ε)·γ+β; x[1,3]ได้ประมาณ[−1,1]ใช้populationvariance1
4. FFNใช้linear d→dff→activation→linear dff→dแยกแต่ละtoken; ผสมfeaturesไม่ผสมตำแหน่งtokenโดยตรง
5. attentionผสมข้อมูลข้ามtoken ส่วนresidualเก็บinputเดิมแล้วบวกoutput; ปิดattentionแต่ยังresidualข้อมูลยังผ่านได้
6. FFNเพิ่มnonlinearfeaturetransformation; ถ้าเอาactivationออกสองlinearรวมได้หนึ่งmatrix จึงต้องablateพร้อมparametercount
7. learnedabsolutepositionที่ยาวเกินtrainไม่มีembeddingหรือgeneralizationไม่แน่; RoPEก็ไม่ได้รับประกันextrapolationทุกlength
8. encoderเห็นทั้งsequenceส่วนdecodercausal; encoder-decoderเพิ่มcrossattentionและsourceencodingcostแต่เหมาะconditionalgeneration
9. X=[[1,0],[0,1]],Q=K=V=X,masknone; Asoftmax(I/√2)ได้diagonal≈.669762,offdiag≈.330238; O=A@X=A
10. assertrowsum1,permutebothinput/outputแล้วselfattentionequivariantเมื่อไม่มีposition; causaluppertriangleต้อง0และpaddingไม่ถูกอ่าน
11. `mu=X.mean(-1,keepdims=True); var=X.var(-1,keepdims=True); Y=(X-mu)/np.sqrt(var+eps)`; X[1,3]ได้±1/sqrt(1+eps)
12. seed7tinyencoderdecodercopytask; คงheads/depthเปลี่ยนFFNactivationทีละตัวและrecordresidual/normalizedtensor,maskedloss,freerunningaccuracy
13. residualต้องsame(B,T,d); concatheads(B,T,h*dk)ต้องprojectกลับd; crossattentionTsource≠Ttargetให้score(B,h,Ttarget,Tsource)
14. LayerNormvarเล็กต้องε>0; pre-LN/post-LNไม่เหมือนกัน,normalizationfp32ช่วยstabilityแต่ต้องเช็คgradient/explodingresidual
15. splitdocumentsก่อนtokenwindows,maskpadding/causalityถูก; checkpointvalเลือกก่อนtest,seq2seqteacherforcedlossกับgenerationmetricแยกรายงาน
16. layerO(BT²d+BTddff),weightsO(d²+ddff),scorememoryO(BhT²); วัดsequenceเล็ก/ใหญ่และแยกembeddingvocabularycost
17. pre-LNblock x→LN→MHA→+x→LN→FFN→residual; decoderเพิ่มcausalselfattentionและcrossattentionตามarchitectureที่ประกาศ
18. torchLayerNormnormalized_shape=d,epsตรงNumPy,γ1β0; copyMHAprojections/FFNและdropout0ก่อนallcloselayeroutput
19. เปลี่ยนpositionnone→sinusoidalบนorder-sensitivecopytask; คงseed/budgetแล้วreportorderedaccuracy; permutationtestควรเสียequivarianceเมื่อpositionผูกตำแหน่ง
20. บันทึกLNorder/eps/masks/position/depth; ทดลองpre-LNลดgradexplosionเทียบpost-LNในdepth8โดยบันทึกgradnormและvalidationqualityหลายseed
## 31
1. patchembeddingตัดimageเป็นpatchP×P×Cแล้วflatten/project; image4×4×1,P2ได้4patches,eachdim4,projection4→d
2. classtokenคือlearnablevectorที่prependเพื่อรวมข้อมูลผ่านattention; เพิ่มtokenจากNเป็นN+1 ไม่ใช่labelจริงใส่เข้าinput
3. positionalembeddingบวกvectorตามpatchindexเพื่อเก็บตำแหน่ง; patchcontentเหมือนกันต่างตำแหน่งจะได้representationต่าง
4. selfattentionให้patchอ่านpatchอื่นจึงglobalreceptivefieldในlayerแรก แต่weightsยังต้องเรียนvisualstructureจากข้อมูล
5. patchembeddingทำทุกpatch ส่วนclstokenเป็นtokenพิเศษหนึ่งตัว; meanpoolpatchแทนclsได้แต่ต้องประเมินใหม่ไม่ถือว่าเหมือนกัน
6. เปลี่ยนpatchหนึ่งอาจกระทบoutputทุกtokenผ่านA; causalmaskปกติไม่เหมาะimageclassificationเพราะจำกัดการเห็นpatchโดยไม่จำเป็น
7. MLPblockใหญ่เพิ่มparametersและoverfitบนข้อมูลน้อย; dropout/weightdecayช่วยแต่ต้องvalidation ไม่อ้างว่าattentionอย่างเดียวพอ
8. Pเล็กเพิ่มN=HW/P²และattentioncostN²; ลดPครึ่งทำN4เท่าและscorememory16เท่าโดยประมาณ
9. image=np.arange(16).reshape(4,4),P2; patchesตามrowmajorเป็น[[0,1,4,5],[2,3,6,7],[8,9,12,13],[10,11,14,15]]
10. assertpatches(4,4),reassembleแล้วได้imageเดิม; H/WหารPไม่ลงตัวต้องrejectหรือpadตามpolicy; clsเพิ่มlengthเป็น5
11. embeddings(B,N+1,d)+positions(1,N+1,d) broadcastbatch; ปรับresolutionต้องinterpolategridpositionแยกclstokenไม่interpolateรวมมัน
12. seed7syntheticquadrantimages; เปรียบP1/P2คงparameterbudgetและlogN,scorebytes,classificationaccuracyจริงรวมการใช้cls/meanpoolที่ระบุ
13. imagechannelsfirstต้องreshape/permuteอย่างถูก patchflattenorderไม่ตรงกับconvprojectionทำผลต่างทั้งที่shapeผ่าน; testsใช้nonuniformpixelsข้อ9
14. LayerNorm/softmaxstableและpixelnormalizationtrainstats; mixedprecisiontestต้องเทียบfp32และตรวจNaNที่patchprojectionก่อน
15. splitตามsourceobjectก่อนaugment; trainrandomcrop,validationresize/cropที่ตรึง; hyperparameters/augmentationไม่เลือกจากtest
16. patchprojectionO(BHW C d),attentionO(BN²d),MLPO(BNddff); reportNรวมclsและpeakactivationmemory
17. image→patchify→linearprojection→prependcls→addposition→transformerblocks→LN→cls/meanpool→classificationhead
18. conv2d(kernel_size=P,stride=P)เทียบpatchflatten@Wเมื่อweightlayoutตรง; copyposition/clsและdropout0ก่อนcomparefullforward
19. เปลี่ยนMLPexpansion2→4คงdepth/heads/data; reportparametercount/accuracy/latencyหลายseedเพราะไม่ได้คงcapacity
20. บันทึกpatchorder/resolution/positioninterpolation/pooling; ทดลองPเล็กเพิ่มsmallpatternrecallภายใต้memorybudgetพร้อมholdoutbackgroundใหม่
## 32
1. normalizationเปลี่ยนรูปข้อความตามpolicyเช่นUnicodeNFC/whitespace; ไม่ควรลบวรรณยุกต์ไทยเพราะเปลี่ยนความหมาย; ต้องversionpolicy
2. tokenizationแบ่งข้อความเป็นหน่วยและmapID; “AI engineer”แบบwhitespaceให้2tokens แต่ภาษาไทยไม่มีเว้นวรรคทุกคำจึงต้องใช้tokenizerเหมาะสม
3. n-gramคือsequenceต่อเนื่องnหน่วย; tokens[a,b,a]ให้bigrams(a,b),(b,a); กำหนดBOS/EOSเพื่อmodelboundary
4. language modelให้P(tokenₜ|prefix); bigramใช้previous1token ส่วนneuralLMเห็นcontextยาวกว่า; probabilityรวม1ต่อcontext
5. normalizationแก้representationก่อนแบ่ง ส่วนtokenizationกำหนดboundaries/IDs; lowercasingทำข้อมูลcaseหายแม้decoderต่อtokensกลับได้
6. LMทำnext-tokenpredictionไม่ใช่ความจริงsemantic; unigramจับfrequencyได้แต่ไม่wordorder จึงต้องholdoutloglossและfailureexamples
7. subwordทำrarewordแบ่งได้แต่sequenceยาวและบางภาษาถูกแบ่งถี่กว่า; bytefallbackช่วยroundtripแต่ไม่ได้รับประกันsemanticquality
8. perplexityขึ้นกับtokenizationและnormalizationจึงเทียบข้ามvocabโดยตรงไม่fair; reportbits/byteหรือtaskmetricร่วม
9. `unicodedata.normalize('NFC', text)` แล้วnormalizewhitespaceตามpolicy; decomposed `e\u0301`ควรเท่ากับé แต่เก็บrawtextเพื่อaudit
10. assertnormalizationidempotent,éกับe+combiningเท่ากัน,NFCไม่เปลี่ยน“ไทย”; emptystringvalidและunknownunicodeไม่หายเงียบ
11. tokens[a,b,a,b], bigramcounts(a,b)=2,(b,a)=1; add-onevocab2ให้P(b|a)=3/4,P(a|a)=1/4; contextunseenใช้priorที่ระบุ
12. seed7splitdocuments fitvocab/countstrain; compareunigram/bigramheldoutNLLและOOVrateด้วยtokenizationเดียว; shuffletrainingorderต้องcountsเท่าเดิม
13. inputIDs(B,T),targets(B,T)shift1; paddinglengthmaskตรงT; ID−1หรือ≥vocabsizeต้องerrorไม่เอาmodvocab
14. probabilityproductsunderflowให้sumlogprobs; smoothingα>0ป้องกันunseengramlog0 แต่ต้องเลือกαvalidationและไม่เพิ่มcountจากtest
15. dedupเอกสารก่อนsplit,fitnormalizerconfiguration/vocabบนtrain; temporalholdoutเมื่อproductiontextเปลี่ยนเวลาและไม่leaknear-duplicates
16. countngramsO(total_tokens·n)โดยstraightforward; dictionarymemoryO(uniquegrams); tokenizerbenchmarkรายงานbytes/secและtokens/byteแยก
17. raw→versionednormalize→tokenize→vocabfittrain→ngrams/counts→conditionalprob→NLL/PPL; generatedsampleต้องdetokenizeตามpolicy
18. compareCounterbigramกับexplicitloopบนtokens[a,b,a,b]ได้countsเหมือนกัน; librarytokenizerต่างกฎต้องไม่อ้างเป็นoracleก่อนalignnormalization
19. คงcorpus/split เปลี่ยนword→subword; reportOOV/tokensperbyte,holdoutbits/byteและtaskmetric ไม่ใช้PPLต่ำสุดข้ามtokenizersตัดสิน
20. บันทึกUnicodeversion/normalization/vocab/specialIDs/split; ทดลองNFCลดduplicateformsโดยThai roundtripไม่เสียและreportcollisioncases
## 33
1. distributionalhypothesisกล่าวว่าคำบริบทคล้ายกันมักความหมายเกี่ยวข้อง; ไม่ได้แปลว่าsynonymเสมอ เช่นhot/coldมีcontextคล้ายแต่ตรงข้าม
2. Word2Vecเรียนwordvectorsด้วยpredictioncontext; CBOWบริบท→center,Skipgramcenter→บริบท; embeddingsเป็นparametersไม่ใช่one-hotoutput
3. CBOWเฉลี่ยcontextembeddingsแล้วทำนายcenter; E(a)=[1,0],E(b)=[0,2]ได้h=[.5,1]สำหรับcontext[a,b]
4. Skipgramใช้centerหนึ่งตัวทำนายcontextหลายคู่; [a,b,c]window1ให้(b,a),(b,c)และboundarypairsตามpolicy
5. hypothesisเป็นแนวคิดเชิงภาษาศาสตร์ ส่วนWord2Vecเป็นobjective/algorithmหนึ่งที่ใช้แนวคิดนี้; cooccurrenceSVDเป็นอีกวิธีหนึ่ง
6. เพิ่มwindowจับsemanticbroadcontextแต่ลดlocal-syntacticfocus; frequentwordsครองpairsต้องsubsamplingโดยประกาศsamplingdistribution
7. negativesamplingต้องใช้Pnegเช่นunigram^.75และห้ามใช้sigmoidlogตรงๆที่overflow; false negativesอาจผลักrelatedwordsออก
8. cosineวัดทิศทางและละnorm; zero-vectorcosineundefinedต้องแจ้ง; similarityสูงไม่ใช่หลักฐานcausal/semanticทั้งหมด
9. สร้างcooccurrencematrixจากtokens[a,b,a,c],window1; counts(a,b)=2,(a,c)=1และsymmetryเมื่อเพิ่มสองทิศ จากนั้นcomparecontextrows
10. assertcountsab2,ac1,boundariesไม่wrapและไม่มีpairข้ามdocument; window0reject; vocabularyIDsคงที่ทำให้matrixrepeatable
11. `h=E[[a,b]].mean(0); logits=h@Wout; p=stable_softmax(logits)`; Eข้อ3,Wout=Iให้p≈[.377541,.622459]
12. seed7tinycorpus trainSkipgramnegativeK2ด้วยfixedpairs; logpositive/negative-dot/NLLและcosineคู่ที่กำหนดก่อนฝึก ไม่เลือกเฉพาะคู่ที่ดูดี
13. embeddinglookup IDs integer(B,contextlen)→(B,contextlen,d); เฉลี่ยaxis1ไม่ใช่axis2; outputvocabdimต้องตรงtargetIDrange
14. ใช้−logσ(pos)−Σlogσ(−neg)=softplus(−pos)+Σsoftplus(neg); dot1000ยังfinite ไม่ใช้exp1000ตรงๆ
15. splitdocumentsก่อนcontextpairs,fitvocab/counts/negdistributiontrain; evaluationintrinsicกับdownstreamholdoutแยกและreportOOVcoverage
16. fullsoftmaxO(BVd),negativesamplingO(B(K+1)d),parametersO(Vd)สองtables; วัดK/batch/vocabและsamplegenerationcostด้วย
17. corpus→tokenIDs→contextpairs→lookupinput/outputE→dots→negative-samplingloss→sparsegradientupdates→evaluation; คำที่ไม่ถูกsampleไม่ควรถูกdenseupdateโดยบังเอิญ
18. copyinput/outputtablesและnegativesเดียวกันเข้าtorch.nn.Embedding; comparesoftpluslossและgradrowsรวมซ้ำเมื่อIDปรากฏหลายครั้ง
19. คงpairs/seed เปลี่ยนnegativeK2→5; reportqualityกับtimeต่อepochและtotaldotproducts ไม่อ้างKสูงดีเสมอหรือcomputeเท่าเดิม
20. บันทึกwindow/dynamicwindow/subsampling/negativepower/seed/vocab; ทดลองwindow2→5ช่วยsemanticretrievalแต่syntacticaccuracyเปลี่ยนอย่างไรโดยfixedtestpairs
