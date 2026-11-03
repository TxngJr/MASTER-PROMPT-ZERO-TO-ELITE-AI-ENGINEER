# เฉลยเฉพาะข้อ: data, training และ preference
## 51
1. document schema ต้องมี document_id, raw_hash, source_uri, license, fetched_at, language, raw_text, normalized_text, normalization_version, split, tokenizer_revision; license/provenance ที่ไม่ทราบต้องบันทึก unknown ไม่เดา
2. ใช้ Unicode NFC และแทน CRLF/CR เป็น LF ตามเวอร์ชัน policy; เก็บ raw bytes ด้วย normalization ต้อง idempotent และไม่ลบวรรณยุกต์ไทยเพียงเพื่อเพิ่ม dedup
3. `digest=hashlib.sha256(normalized.encode('utf-8')).hexdigest()`; ใช้ set ของ digest เพื่อตรวจ exact duplicate และตรวจ policy normalization ก่อน hash; “a” กับ “b” ต้องได้คนละ digest
4. shingles2ของ[a,b,c]เป็น{ab,bc},[a,b,d]เป็น{ab,bd}; Jaccard=1/3; emptysetทั้งคู่ต้องกำหนดผล ไม่หารศูนย์
5. splitจาก hash(document_group_id+split_version) modulo100 เช่น <80 train,<90 validation,ที่เหลือtest; near-duplicate group ต้องทั้งกลุ่มอยู่ split เดียว
6. โหลด tokenizer checkpoint/revision ที่ freeze แล้ว encode train/validation/test ด้วย mapping เดียว; ห้าม fit merges ใหม่ใน validation และ assert IDs ใน vocab
7. docs[a,b]และ[c]เมื่อ EOS=9,block4 ได้[a,b,9,c]แล้ว[9,...]ตาม remainder policy; packed attention จะอนุญาตข้ามเอกสารหรือไม่ต้องประกาศ
8. sort stable document IDs, fixed shard size, fixed serialization/newline แล้วเขียน SHA-256 แต่ละ shard; rerun input/config เดิมต้องได้ bytes/digests เดิม
9. sourcecounts[100,400],temperatureT2ให้ weights proportional n^(1/T)=[10,20]จึง probabilities[1/3,2/3]; T1คืนสัดส่วน1/5,4/5
10. manifest รวม source hashes/counts/licenses/split hashes/tokenizer/packing/shardchecksums/configversion; validator ต้องอ่านไฟล์จริง ตรวจ counts และไม่รับ checksum หาย
11. Jaccard ของข้อ4ได้1/3; duplicate group ที่ threshold.3 ต้องรวมกันก่อน split ไม่ใช้ label ของตัวอย่างช่วยตัดสินความคล้าย
12. normalize twice เท่าครั้งเดียว, exact duplicate ไม่ข้ามsplit, decode-roundtripตามtokenizerpolicy และ shard bytes/checksums identical หลัง rerun
13. เปรียบ near-dedup threshold.7/.9 คงcorpus/normalization/seed; วัดremovedtokens,cross-split overlapและheldoutquality ไม่ลบทิ้งมากแล้วอ้างดีขึ้นอัตโนมัติ
14. record raw/normalizedchars,tokenIDs(N,),packed(B,T),valid-token counts,source mixture และ validationNLL แยกภาษา
15. near-duplicatesข้ามsplitหรือfit tokenizerบนtestทำheldoutดูดีเกิน; provenance audit ต้องแสดงgroup IDs และ leakage checks ไม่ใช้เพียงrandomseed
16. exact hash O(totalbytes), pairwise Jaccard naive O(N²·shingles); MinHash/LSHลดcandidate workแต่ต้องวัดrecallของdedupและmemory
17. SHA-256เทียบ hashlib fixture กับ known digest, packingเทียบexplicitconcatenation; near-dedupเทียบexhaustiveJaccardบนtinycorpusก่อนใช้LSH
18. เปลี่ยน mixture temperature1→2 คงsources/tokenbudget; reportactualsampledsourcecountsและperlanguagevalNLLหลายseed
19. manifest มีcodecommit/rawcorpushashes/normalization/dedup/split/tokenizerversion/packing/sourceweights/shards/RNG และcommandที่สร้างซ้ำ
20. ทดลอง“near-dedupลดcross-split overlapเหลือ0โดยtokensเหลือ≥80%”; exactdedupbaseline,fixedcorpusและmanuallyreviewfalsepositivepairs
## 52
1. strided IDs rank r รับ range(r,103,8); ranks0..6ได้13ตัว,rank7ได้12 รวม103และไม่ซ้ำ; DDP training ต้องกำหนดpad/dropเพื่อstepsเท่ากัน
2. globalbatch=Bmicro·accum·worldsize; B2,K4,W8ได้64samples/update,T128ได้8192tokensหากไม่มีpadding
3. gradients[1,3,5,7]จาก4ranksเฉลี่ยได้4; DDPsumแล้วหารworldsize ต่างจากsum16และต้องคุมlossreductionต่อrankเมื่อbatchsizesไม่เท่ากัน
4. ringallreduceแต่ละrankส่งและรับประมาณ2(W−1)/W·S; W8,S2GBได้3.5GBต่อrankต่อทิศตามนิยามtrafficนี้ ไม่ใช่2GB/8
5. ให้weightsP,gradsG,optimizerO; stage0=P+G+O,stage1=P+G+O/W,stage2=P+(G+O)/W,stage3=(P+G+O)/Wก่อนcommunicationbuffers/activations
6. simplified pipeline efficiency=m/(m+p−1); p4,m1=.25,m4=4/7,m16=16/19; scheduleจริงมีcommunication/imbalanceเพิ่ม
7. data parallel ทำmodelสำเนาคนละbatch; tensor parallel แบ่งmatrixในlayer; pipeline parallel แบ่งlayersระหว่างstages; hybridต้องแยกprocessgroups
8. rankหนึ่งหมดdataก่อนทำcollective แต่อีกrankยังallreduceจะรอ; pad/drop/joinprotocolและทุกrankเรียกcollectiveลำดับเดียวกัน ห้ามแก้ด้วยtimeoutอย่างเดียว
9. checkpointmetadata มีworldsize/rank/shardlayout/globalstep/optimizerpartition/RNG/samplerposition/architectureversion; reshardต้องmapglobalparameterIDsไม่อิงrankfilenameอย่างเดียว
10. speedup=T1/TW,efficiency=speedup/W; T1=80s,T4=25sได้speedup3.2,efficiency.8; keepglobalworkfixedสำหรับstrongscaling
11. rankmean[2,4]ที่batchsizes1,3ต้องweightedmean3.5 ไม่ใช่3; lossnormalizationต้องสอดคล้องglobalvalidtokens
12. assertshardunionครบ,overlapตามpadpolicyเท่านั้น,rankstepsเท่ากันและaveragedgradientตรงsingleprocessglobalbatchบนtinylinearfixture
13. worldsize1/2/4 คงglobalbatch/model/totalsteps; reportthroughput/scalingefficiencyและlossnumericaltoleranceไม่เปลี่ยนLRพร้อมworldsizeโดยไม่บอก
14. recordlocalbatch/ranksteps/globaltokens,gradientbucketbytes/communicationtime/peakstatebytesและvalidationlossที่aggregatesweightedถูก
15. duplicatevalidationจากsamplerpaddingทำmetricbias; gatheruniqueIDsหรือweightdedup ไม่เฉลี่ยrankmetricsแบบunweightedเมื่อcountsต่าง
16. DPcomputeประมาณsinglemodel/Wเมื่อglobalbatchแบ่งได้แต่communicationO(P)ต่อstep; stage3ลดstateแต่เพิ่มallgather,activationไม่รวมในสูตรstate
17. CPUtoyallreducereferenceเทียบmanualweightedgradient; GPU/NCCLthroughputต้องแยกสถานะnot_runหากไม่มีhardware ไม่ใช้CPU arithmeticยืนยันdistributedGPU
18. เปลี่ยนbucket sizeคงworldsize/topology/globalbatch; measurecompute-overlap/commtimeและpeakmemoryหลายrunsหลังwarmup
19. manifest มีtopology/processgroups/worldsize/globalbatch/samplerpad/precision/collectivebackend/versions/checkpointlayout/network/hardware/seeds
20. ทดลอง“accumulation4ลดcommunicationperprocessedtokenโดยvalNLLคงในtolerance”; baselineaccum1ด้วยglobalbatchเท่ากันและfixedtokenbudget
## 53
1. grid=ceil(N/blocksize); N1000,block256ได้4blocks; N0ให้0และไม่launchinvalidgrid; lastblockต้องboundcheck
2. globalindex=blockIdx.x*blockDim.x+threadIdx.x; block2,block256,thread7ให้519; index≥Nห้ามอ่าน/เขียน
3. thread0..255: warp_id=threadIdx//32,lane=threadIdx%32; thread77อยู่warp2,lane13 และมี8warpsต่อblock
4. 32lanesfloat32contiguousอ่าน128bytesต่อalignedspan; stride32กระจายหลายsegmentsจึงtransactionsมากขึ้น ต้องระบุsegment/alignment assumptions
5. registersเป็นthread-local,sharedmemoryเป็นblock-sharedต้องsync,globalmemoryข้ามblocks/device-wide; sharedไม่ใช้สื่อสารระหว่างblocksโดยไม่มีglobalprotocol
6. limits threads2048/SM,registers65536,shared64KiB; block256,regs32/thread,shared16KiB ให้min(8,8,4)=4blocks/SMก่อนarchitecturalblocklimit
7. vectoraddหนึ่งelementอ่าน2เขียน1float32=12bytes,1FLOPให้AI1/12; matmuln³multiply-add≈2n³FLOPsและขั้นต่ำ3n²·4bytesให้AI≈n/6หากreuseสมบูรณ์
8. rooflinebound=min(peakFLOPs,bandwidth·AI); peak10TF/s,BW500GB/s,AI2ให้bound1TF/s จึงmemoryboundตามsimplifiedmodel
9. CUDAbenchmarkwarmup,torch.cuda.synchronizeก่อน/หลังtimingหรือCUDAeventsแล้วelapsed_time; CPUenqueue timeไม่ใช่kernelruntime
10. แยกH2D copy,D2H copy,kernelและend-to-end; pinnedmemory/nonblockingอาจoverlapเมื่อstream/dependencyพร้อม ต้องmeasureจริงไม่อ้างlatencyลดเสมอ
11. N1000/block256มี1024threadsแต่valid1000;24threadsสุดท้ายต้องไม่access การเพิ่มgridโดยไม่boundscheckไม่ใช่performanceoptimization
12. CPUindexsimulationassertvalidindices0..N−1ครบครั้งเดียวและinvalidguard; GPUkerneltestต้องcompareoutputCPUoracleรวมNไม่หารblockลงตัว
13. block128/256/512คงN/kernel/input/device; reportruntime/occupancyestimate/registeruseจริง ไม่เลือกblockใหญ่ที่สุดจากสมมติฐานอย่างเดียว
14. recordgrid/block/warps/validthreads,bytesmoved,FLOPs,elapsedmsและmaxoutputerror; hardwareGPUstatusแยกจากCPUformula
15. นับenqueue timeหรือรวมfirst-JITcompileกับบางconfigurationทำbenchmarkผิด; warmup/sync/repetitionsและsameclock/powerstateเท่าที่ควบคุมได้
16. vectoraddO(N)เวลา/bytes,matrixmultiplyO(n³)FLOPs; memorylayout/reuseทำeffectivecostต่าง ต้องprofilebeforeoptimization
17. indexและvectoraddเทียบNumPyCPUfixtures; actualCUDAopเทียบtorchCPUfloat32within tolerance,hardwareไม่มีต้องรายงานnot_runไม่skipเป็นผ่าน
18. contiguous→stridedaccessคงจำนวนusefulelements/dtype; measuretransactions/latencyด้วยprofilerเมื่อGPUมี และcheckoutputเหมือนกัน
19. manifest มีGPUmodel/driver/CUDA/runtime/kernelcode/launchconfig/N/dtype/warmup/repeats/sync/copyinclusion/powerstate/seed
20. ทดลอง“tilingลดglobalbytesต่อmatmulและเพิ่มGFLOPs≥1.5×”; naivebaseline,samedim/device,error≤toleranceพร้อมshared-memorybudget
## 54
1. FP32มีsign1/exponent8/fraction23,FP16=1/5/10,BF16=1/8/7; BF16rangeใกล้FP32แต่precisionหยาบกว่าFP16
2. 1Bparameters rawFP32=4GB,FP16/BF16=2GBแบบdecimal ไม่รวมgrad/moments/masterweights/activation/allocator
3. np.float16(1e5)overflowเป็นinf,1e−9อาจเป็น0; maxfinite65504,minnormal≈6.10e−5,minsubnormal≈5.96e−8; subnormalbehaviorขึ้นกับhardware
4. BF16roundingเก็บhigh16bitsของfloat32พร้อมround-to-nearest-even; comparetorchfloat32.to(bfloat16).float และrelativeerrorเฉพาะnonzeroreference
5. g=1e−8,scale65536ได้scaledg.00065536; คำนวณในprecisionที่เก็บได้แล้วunscaleคืน1e−8ก่อนstep; scaleไม่ทำtruegradientใหญ่ขึ้น
6. พบnonfinitegrad→skipoptimizerstep,scale*=.5; finiteครบgrowthinterval→scale*=2; growthtrackerและcompletedupdatesต้องแยก
7. clipscaledgradก่อนunscaleทำeffectiveclipthresholdผิด; gnorm4,scale8,maxnorm2: ต้องunscale32→4แล้วclip2 ไม่clip32→2แล้วunscale.25
8. `with torch.amp.autocast('cuda',dtype=torch.float16): loss=...`; scaler.scale(loss).backward(); scaler.unscale_(opt); clip; scaler.step(opt); scaler.update(); CUDAต้องมีhardwareจริง
9. CPUใช้torch.amp.autocast('cpu',dtype=torch.bfloat16)กับLinearเล็ก,backwardlossfinite,compareFP32tolerance; ไม่ใช้CUDAFP16GradScalerเพื่ออ้างCPUtestเทียบGPU
10. คงmodel/batch/sequence/optimizer/warmup; วัดpeakallocated/tokenssecและmaxerror/valNLLแยกprecision CPUผ่านยังไม่พิสูจน์GPUthroughput
11. fp16rawweights2GBแต่AdamFP32moments8GBต่อ1Bและgrads/masterweightsเพิ่มอีก; precisionweightsไม่เท่ากับtrainingmemoryทั้งหมด
12. nonfinitegradต้องไม่เปลี่ยนweights/updates,unscale-before-clipfixtureถูก,CPU-BF16lossfinite; GPUrequiredtestถ้าไม่มีhardwareต้องnot_runชัดเจน
13. compareFP32/BF16บนCPUdeviceเดียวก่อนmeasureGPUแยก; recordoperatorfallback/dtypeจริงไม่อ้างทุกopBF16เพียงอยู่autocastcontext
14. recordweight/activation/grad/momentdtypes,scale,growthtracker,skippedsteps,maxerrorและvalNLL; monitorinfก่อนoptimizerstateเสีย
15. ต่างbatchsizeหรือใช้FP32coldstartเทียบmixedwarmcacheทำthroughputดูดีเกิน; freezebenchmarkและvalidatequalityก่อนสรุป
16. storageO(P)เปลี่ยนbytesperelement,computeขึ้นกับsupportedkernels; autocastconversionมีoverheadและactivationoften dominates ไม่คาดspeedup2×เสมอ
17. NumPyfloat16castเทียบTorchCPUhalfcastบนrepresentablevalues; BF16เทียบbitroundingoracleและties-to-evenboundaryรวมNaN/infpolicy
18. scale1→dynamicคงmodel/data/seed/precision; reportunderflow/skippedstepfrequency/valqualityหลายruns ไม่เปลี่ยนLRพร้อมกัน
19. manifest มีhardware/precision/autocastoplist/masterstate/scalerconfig/cliporder/batch/context/versions/seeds/error-tolerance
20. ทดลอง“BF16ลดnonfiniteupdatesเทียบFP16บนlargeactivationtoyโดยNLLไม่แย่”; samehardware/model/budgetและreportFP32reference
## 55
1. backbone9900params,head100ให้trainableratio100/10000=.01=1%; numeratorเฉพาะrequires_gradและdenominatorรวมทุกparameterไม่activation
2. ตั้งbackbone.requires_grad_(False),optimizerเฉพาะhead; loss.backwardแล้วassertbackbonegradNoneและweightsก่อน/หลังallclose ส่วนheadเปลี่ยน
3. unfreezeเฉพาะlast2blocksและเพิ่มparametersเข้าoptimizer/paramgroups; ถ้าoptimizerสร้างก่อนfreeze/unfreezeต้องตรวจรายชื่อไม่ให้trainableหลุด
4. optimizerparamgroupsbackbonelr1e−5,headlr1e−3; groupIDsไม่ซ้ำ/ไม่ตกหล่น,decaypolicyระบุbias/normแยก
5. pretrainedกับscratchเริ่มคนละweightsแต่architecture/data/budget/splitsเหมือนกัน; reportlearningcurves/targetqualityและcomputeไม่อ้างpretrainedดีทุกdomain
6. checkpointเลือกvalmetricสูงสุดหรือvallossต่ำสุดตามที่ประกาศ; savebestthenloadก่อนtest ไม่ใช้lastโดยเรียกbest
7. earlystoppingpatience2: val[.5,.4,.41,.42]หยุดหลังสองnonimprovementsและrestoreepoch2; min_delta/tiesต้องกำหนด
8. targetgain=targetmetric_after−before,retention=generalmetric_after−before; reportทั้งคู่เพื่อเห็นcatastrophicforgetting ไม่ใช่targetgainอย่างเดียว
9. groupIDuser/documentfamilyหนึ่งอยู่splitเดียว; assertset(group_train)disjoint(group_test),near-duplicatesก่อนsplit
10. fullFTกับfrozenbackboneใช้samepretrainedcheckpoint/split/seed; reportquality/trainableparams/peakmemory/latencyและ3seeds ไม่ให้headbaselineงบน้อยกว่าโดยไม่บอก
11. 100trainableจาก10000=1%; ถ้าunfreezeblock200เพิ่มtrainableเป็น300=3%และoptimizerต้องมีทั้ง300จริง
12. frozenweightsunchanged,trainableparametersมีgradและcheckpointroundtripvalid; groupdisjointและbestcheckpointselectedfromvalเท่านั้น
13. head-onlyเทียบlast2blocksคงdata/budget/init; reporttargetmacroF1/retention/peakbytesพร้อมparametercount
14. recordbackboneoutput(B,d),headlogits(B,C),trainableIDs,gradnormbygroupและtarget/generalmetrics; dropoutmodeของfrozenbackboneต้องชัด
15. evaluationซ้ำข้อมูลpretrainingหรือuserข้ามsplitทำtargetgainสูงเกิน; auditdedup/sourcegroupsและไม่tuneonretentiontest
16. frozenbackboneยังมีforwardcost; no_gradลดactivationmemoryถ้าไม่ต้องgradientthrough ส่วนfullFTเก็บbackwardactivationsและoptimizerstateO(trainableP)
17. comparemanualheadlinearforward/gradกับframeworkด้วยcopiedweights; frozenbackbonetestคือparameterdiff0ไม่ใช้lossลดอย่างเดียว
18. discriminativelrbackbone=head/100เทียบsameLRคงgroups/seed; reporttargetgainและforgettingหลายruns
19. manifest มีbasecheckpointrevision/datasetsplits/frozenlayerIDs/paramgroups/LRs/decay/earlystop/seed/retentionbenchmark/modeldigest
20. ทดลอง“last2blocksเพิ่มtargetF1≥.03โดยretentionลด≤.01เทียบhead-only”; fixedbudget/pretrainedbaseและpairedholdoutCI
## 56
1. r(din+dout)=8192r; r4/8/16/64ให้32768/65536/131072/524288 เทียบbase16777216parameters ไม่รวมbias/scaling
2. conventionW(dout,din),A(r,din),B(dout,r),ΔW=(α/r)BA; inputx(din),outputWx+(α/r)B(Ax)
3. B0ทำBA0แม้Aสุ่ม จึงinitialoutputเท่าbase; firstgradientdAอาจ0แต่dBไม่จำเป็น0 ต้องไม่ตั้งA/Bทั้งคู่0จนgradientติด
4. ratio=r(din+dout)/(din·dout+r(din+dout))หากdenominatorรวมadapter; รายงานbase-onlyratioแยกไม่สลับนิยาม
5. mergedW=W+sBAแล้วforwardต้องallcloseunmerged; unmergeลบsameΔWครั้งเดียว,ห้ามmergeซ้ำและอย่าแก้A/Bขณะถือmergedweightsโดยไม่track
6. targetnamedmodulesq_proj/v_projในtinyTransformer,wrapLinearให้basefrozenและA/Bregistered; asserttrainableIDsเฉพาะadapters
7. q/v-onlyมีfewertrainablesส่วนall-linearรวมFFN/O/Kเพิ่มcapacityและmemory; samebase/split/budgetcomparequalityและparamsไม่สรุปallดีกว่าแน่
8. AdamFP32moments2·4bytesต่อtrainableparam; fullFTmoments8P,LoRA8r(din+dout)แต่baseweightsและforwardactivationsยังต้องอยู่
9. codebook16levelsระบุค่าชัดเช่นlinspace(−1,1,16),scale=maxabsW; nearestcode→dequant,reportMSE; generic16levelไม่เท่ากับNF4เสมอ
10. metadataต้องpinbaserepo/revision/hash,architecture,targetmodules,r,α,dropout,dtype/quantization/tokenizer; sameชื่อน้ำหนักแต่revisionต่างต้องreject
11. din=dout2,r1,α1,A=[1,0],B=[2,3]ᵀให้ΔW=[[2,0],[3,0]]; x[1,2]adapteroutput[2,3]
12. B0no-op,merged/unmergedallclose,frozenbaseunchanged,metadatawrongbasefailsและshapesrconsistent; toleranceตามdtypeหลังmerge
13. rank4/8/16คงbase/targets/α/rpolicy/seed/updates; reportquality/trainableparams/memoryและไม่เปลี่ยnscalingโดยไม่ระบุ
14. recordx(B,din),Ax(B,r),BAx(B,dout),base/adapteroutput,gradnormA/BและvalNLL; basegradientต้องNone
15. ใช้adapterกับbaseคนละrevisionแม้shapeเท่ากันอาจผลเสีย; evaluationต้องpinbase/quantconfigและfreezeheldoutไม่เลือกbestseed
16. adapterforwardextraO(BTr(din+dout)),stateO(r(din+dout)); mergedinferenceไม่เพิ่มlowrankmatmulsแต่quantizedmergeอาจเพิ่มroundingerror
17. NumPyWx+sB(Ax)เทียบTorchLinearwrapperfloat64copiedweights; PEFToptionalintegrationต้องเทียบorientation/fan_in_fan_out/scalingด้วย
18. q/v→alllinearคงrank/data/initseed; reportquality-per-trainableparameterและlatency/memoryหลายruns ไม่อ้างparameterbudgetเท่ากัน
19. manifest มีbasehash/tokenizer/targetIDs/r/α/scaling/init/quantcodebook/compute-dtype/seed/trainingdatahash/adapterdigest
20. ทดลอง“rank8คงquality≥99%ของrank16แต่optimizerbytesครึ่ง”; samebase/task/budget,fixedholdoutmetricและ3seeds
## 57
1. instruction “สรุป”,input “เนื้อหา”,output “คำตอบ”→messages[{role:user,content:'สรุป\nเนื้อหา'},{role:assistant,content:'คำตอบ'}]; emptymissinginputpolicyชัด
2. render `<user>...<end><assistant>...<end>`ตามtemplateversionที่modelใช้; escape/encodecontentไม่ให้ข้อความผู้ใช้ปลอมrolemarkerได้
3. labels=inputIDs.copy(); positionsuser/system/pad→−100,assistantcontent+EOSตามpolicyคงID; maskต้องตามtokenoffsetจริงไม่เดาจำนวนcharacters
4. logitspositiontทำนายlabelt+1; assistantfirsttokenที่indexkใช้logitsk−1; ชิฟต์labelsกับmaskพร้อมกันเพื่อไม่superviseuserท้ายturnผิด
5. multi-turnA1/A2ต้องmaskbothassistantspansรวมEOSแต่userturnsไม่ถูกloss; testช่วงassistantก่อนหน้าที่ยังอยู่หลังtruncationไม่ถูกลืม
6. fullsequenceรวมlossuser+assistant ส่วนassistant-onlyลดdenominatorเฉพาะassistant; ค่าต่างไม่บอกqualityโดยตรงเพราะobjectiveต่างกัน
7. conversation1/2คั่นEOS/boundaryและระบุattentionblockdiagonalหรือcrossconversationallowed; labelsไม่ไหลข้ามboundaryและpositionresetตามpolicy
8. truncateแล้วvalidlabelcount0ต้องreject/skipพร้อมcounterreason ไม่lossmeanemptyหรือpretendbatchผ่าน; ควรtruncateuserก่อนassistantตามpolicy
9. exacthashcanonicalmessagesและnear-dupgroupก่อนsplit; paraphrasesหรือsharedresponseleakต้องaudittextpairsไม่ใช้เฉพาะrawJSONhash
10. heldoutsetมีunseentasks/constraints/multiturnและrubricpassfailเฉพาะtask; เก็บexpectedformat/groundtruth/forbiddenclaimsและไม่ใช้trainexamplesซ้ำ
11. IDs[user10,assistant20,EOS2],assistantmask[0,1,1]หลังshifttargets[20,2]ทั้งสองvalid; targetuser10ไม่ถูกทำนายจากlogitsตำแหน่ง−1
12. assistantmaskตรงtokenoffset,causalshiftถูก,zero-supervisedrejectและusertextcontainingrolemarkerไม่เปลี่ยนrolestructure
13. fullsequence/assistant-onlyคงtokens/model/seed/updates; reportheldoutinstructionadherenceและassistantNLLแยกจากtotalloss
14. recordrenderedtext/tokenIDs/rolespans/labels(B,T)/validcounts/logits(B,T,V)และformat-complianceจากgenerationจริง
15. ใช้sameinstructiontemplate/answernear-duplicatesข้ามsplitหรือjudgeเห็นmodelnameทำคะแนนbias; freezeevalและblindgrading
16. maskconstructionO(tokens),packinglinearแต่trainingattentionO(T²d); assistant-onlylossไม่ลดforwardattentioncostถ้ายังเห็นทั้งsequence
17. handcomputedshift/maskfixtureเทียบTorchCEignore_index−100; templateจากtokenizerต้องsnapshotversionและcompareexpectedIDsไม่ใช่stringcontainsassistant
18. เพิ่มmulti-turnsupervisionแทนlast-turn-onlyคงconversations/tokenbudget; reportearlier/laterturnaccuracyและvalidtokenfractionหลายseed
19. manifest มีchattemplate/rolespecialIDs/EOSpolicy/assistantmask/truncation/packing/dedup/splits/basecheckpoint/seed/evalrubricversion
20. ทดลอง“assistant-onlyช่วยformatcompliance≥5จุดเทียบfullsequenceโดยfactualaccuracyลด≤1จุด”; fixedtrainingtokensและblindholdout
## 58
1. Bradley-TerryP(chosen)=σ(rc−rr); rc2,rr1ให้σ1=.731059; เพิ่มrewardconstantทั้งคู่probคงเดิม
2. pairwiseloss=softplus(−(rc−rr)); margin1ได้.313262,margin−1ได้1.313262; ใช้stablelogaddexpไม่logsigmoidตรงๆ
3. accuracy=mean(rc>rr)ตามtiepolicy; margins[1,−.2,0]ให้1/3ถ้าtieไม่ผ่าน; reporttiesแยกไม่สลับlabelเพื่อเพิ่มคะแนน
4. KL(P||Q)=ΣPlog(P/Q); P[.8,.2],Q[.5,.5]ได้≈.19274476; supportQ0แต่P>0ให้infiniteต้องerror/explicitpolicy
5. sampledaction1เมื่อP.2,Q.5ให้log(.2/.5)=−.916291 แม้exactKL.192745≥0; singlelogratioไม่ใช่exactKL
6. advantages[1,3]mean2,popstd1→[−1,1]; std0ใช้epsilonหรือreturnzeroโดยประกาศและไม่normalizeacrossunrelatedpadding
7. ratio=exp(logπnew−logπold); oldprob.2,new.3ให้1.5; PPOoldpolicyไม่ใช่referencemodelที่ใช้KLregularization
8. ε.2,A2,ratio1.3ให้surrogate2.4; A−2,ratio.7ให้−1.6; minimizeลบsurrogatemeanและclipratioไม่clipadvantage
9. shapedreward=rm_score−β(logπ−logπref); score1,β.1,logratio.5ได้.95; sampledpenaltyอาจติดลบต้องแยกexpectedKL
10. independentcasesมีrewardproxyสูงแต่wronganswer/verbosity/spuriousformat; blindhumanrubricประเมินtaskcorrectnessไม่ใช้rewardmodelตัวเดิมเป็นsolejudge
11. rmchosen2,rejected1loss.313262; policyKL.192745ไม่เท่ากับsampledlogratioทุกsample; ตัวอย่างนี้จับการรายงานKLผิดนิยาม
12. rewardshiftinvariance,exactKL≥−tolerance,KL(P||P)=0,clipfixturetwo-signsถูกและreference/oldpolicyfrozen
13. β.01/.1คงpolicy/reference/prompts/rewardmodel/budget; reporttaskmetric/reward/trueKLแยกและหลายseedเพื่อตรวจrewardhacking
14. recordcompletionmask/tokenlogprobs/rewardscalar/advantages/ratio/clipfraction/exacttoyKLและindependentaccuracy
15. rewardmodeltrain/evalpairsซ้ำหรือใช้RMscoreเป็นqualityอย่างเดียวทำวงจรself-confirming; preferenceholdoutและindependenttaskjudgeแยก
16. policy+reference+reward+valueforwardเพิ่มcompute/memory; PPOrolloutstorageO(BT)ต่อtokenstatsนอกเหนือmodelstates ต้องวัดทุกmodel
17. stablepairlossNumPyเทียบtorchsoftplusและKLกับtorch.distributions.kl_divergenceCategorical; orientationP||Qต้องตรง
18. advantage-normalizationon/offคงrollouts/seed/LR; reportgradnorm/clipfraction/taskqualityและไม่เปลี่ยนrewardscaleพร้อมกัน
19. manifest มีpolicy/reference/old-policy/RM/valueversions/prompthash/preferencehash/β/PPOclip/rolloutseed/rewardnormalization/evalrubric
20. ทดลอง“KLpenaltyลดindependentfailureแม้RMscoreอาจลด”; β0baseline→.1,fixedrolloutbudgetและblindtaskscore/lengthslices
## 59
1. completionlogp=Σtmaskt·logp(yt|prefix); logps[−.1,−.2,−.3],mask[0,1,1]ได้−.5; padding/promptไม่รวมและstandardDPOใช้sumไม่mean
2. policymargin=logπ(chosen)−logπ(rejected); −2−(−5)=3 ใช้completionเดียวกันกับreferenceไม่รวมpromptscore
3. referencemargin=logπref(chosen)−logπref(rejected); −3−(−4)=1; referencefreezeและtokenizer/templateต้องตรงpolicy
4. z=β(policy_margin−reference_margin); β.1,margins3/1ให้z.2,losssoftplus(−.2)=.5981389; ไม่ใช้βคูณlogsigmoidข้างนอก
5. marginadvantage2,β.1/.5ให้loss.598139/.313262; negativeadvantageเพิ่มβทำlossเพิ่มและgradientscaleเปลี่ยน ไม่รับประกันtrainingqualityเพิ่ม
6. implicitpreferenceaccuracy=mean(policy_margin−reference_margin>0)ตามDPOmetricที่ประกาศ; chosenpolicyhigherprobaccuracyเป็นอีกmetricห้ามสลับ
7. reportchosen/rejectedtokenlengthmeans/distributionsและlength-conditionedaccuracy; summedlogpมีlengthdependenceแต่ไม่แก้เป็นmeanเงียบๆเพราะเปลี่ยnobjective
8. cachekeyhash(referenceweights/tokenizer/template/datasetcompletionIDs/maskpolicy/reduction); อันใดเปลี่ยนต้องinvalidate ไม่ใช้เพียงexampleID
9. educationalIPO loss=(policy_margin−reference_margin−1/(2β))²; β.5,advantage2,target1ให้loss1; ระบุสูตรนี้แยกจากvariantsที่length-normalize
10. DPOใช้preferencepairs+frozenreferenceฝึกpolicyตรง; RM/PPOต้องrewardmodelและonpolicyrollouts/value/oldpolicy เพิ่มcomponentsแต่สำรวจใหม่ได้
11. β.1,advantage2ให้z.2,loss.598138869; derivativeต่อpolicymargin=−βσ(−z)≈−.0450166 จึงupdateเพิ่มchosen-relative-margin
12. policy=referenceให้losslog2,gradientfinite; swapchosen/rejectedเปลี่ยนsignz,maskedpromptlogpไม่กระทบlossและreferenceไม่มีgrad
13. β.1/.3คงpairs/base/ref/seed/tokenbudget; reportheldoutpreferenceaccuracy/taskquality/KL/lengthbiasไม่เลือกจากtraininglossคนละβ
14. recordchosen/rejectedlogps,completioncounts,margins,z/loss,gradnormและheldoutwinsแยกlengthslices
15. judgeชอบคำตอบยาวหรือpreferencepairsnear-duplicateข้ามsplitทำwinrateสูงปลอม; blindanswerorderและbalancedlength-eval
16. DPOforwardpolicy2responses+reference2responsesหากไม่cache; cacheลดreferencecomputeแต่storageO(numberpairs),trainingactivationตามcompletionlength
17. NumPysoftplusDPOเทียบTorchF.logsigmoidด้วยsamecompletionmask/sumreduction; finite-differencegradientกับβ/marginตรง
18. referencecacheoff/onคงweights/pairs/templates; assertloss/gradallcloseและmeasurewalltime/cachebytes ไม่ใช้stalevaluesเพื่อเร่ง
19. manifest มีpolicy/refdigest/tokenizer/chattemplate/pairIDs/chosen-labelprovenance/β/logpreduction/mask/cachekey/seed/evalset
20. ทดลอง“length-balancedpairsลดlong-answerbiasโดยheldoutaccuracyไม่ลด”; originalpairsbaselineและlength-controlledblindjudgingหลายseed
## 60
1. normalizationversionเช่นNFC+casefold+collapsewhitespaceต้องsnapshot; ไม่ลบเครื่องหมายที่มีความหมายในโจทย์math/code และเก็บrawanswerเพื่อaudit
2. goldtokens[a,b,c],pred[a,b,d]intersectionmultiset2ให้precision=recall=F1=2/3,EM0; duplicatesใช้Counterintersectionไม่set
3. candidateanswerlogp=sumcompletiontokenlogpsโดยpromptmaskไม่รวม; candidatesต่างlengthอาจbias ให้กำหนดsum/lengthnormตามbenchmarkก่อนดูผล
4. binaryBriermean((p−y)²); p[.8,.3],y[1,0]ได้.065; ECEbinsนิยามconfidence/accuracy/edgesและweightsตามcountsไม่เฉลี่ยbinsเท่าๆกัน
5. seededrandomswapA/Bต่อpair,recordpermutationและremapjudgechoiceกลับmodelID; judgeไม่เห็นชื่อmodelและauditpositionbiasด้วยswapbothorders
6. bootstrapสุ่มindicesnตัวแบบreplacementBครั้งคำนวณaccuracyทุกreplicateแล้วpercentile2.5/97.5; seed/quantilemethodชัดเจน ไม่ใช้Wilsonแทนแล้วเรียกbootstrap
7. pairedbootstrapสุ่มexampleindicesเดียวกันให้modelsA/B,คำนวณΔaccuracyแต่ละครั้งแล้วCIของΔ; independentresamplingทำลายpairingและvarianceต่าง
8. normalizeตามpolicyแล้วSHA-256เทียบtrain/evalexactmatch พร้อมsourceIDs; absenceofexactmatchยังไม่พิสูจน์ไม่มีnear/paraphrasecontamination
9. sliceslanguageไทย/อังกฤษ,taskmath/code/QA,lengthshort/longกำหนดก่อนeval; reportn/metric/CIแต่ละsliceและไม่เปิดเผยข้อมูลกลุ่มเล็กผิดpolicy
10. manifestมีmodel/tokenizer/datahash/prompt/template/decoding/seed/normalization/scorer/judgeversion/hardware/testdate/per-sampleoutputs
11. successes8จาก10accuracy.8; bootstrapreplicateindices[0,0,1,2,3,4,5,6,8,9]บน[1×8,0,0]ให้accuracy.8จริงจากresample ไม่กำหนดCIปลอม
12. perfectpredictionBrier0,F1emptycasesตามpolicy,pairedidenticalmodelsΔทุกreplicate0และbootstrapseedrerunidentical; reordermodelanswerไม่เปลี่ยนmappedscore
13. sumlogpเทียบlengthnormคงcandidateIDs/logits; reportanswerlengthsliceaccuracyและbenchmarkprotocol ไม่choosebestnormalizationหลังtest
14. recordraw/normalizedanswers,completionmask/logps,peritemscore,sliceIDs/bootstrapindices/ΔdistributionและaggregateCI
15. tuneprompt/testnormalizationจากtestหรือjudgeรู้modelnameทำbias; freezescorerversionและholdoutใหม่ พร้อมmanualauditsampleblind
16. scoringtextO(totalcharacters/tokens),bootstrapO(BN)เวลาถ้าmaterializeindicesใช้O(BN)memory; streamingreplicatesลดmemoryแต่ต้องเก็บseed/config
17. bootstrapquantilesเทียบNumPydefaultlinearบนreplicates/indicesเดียว; EM/F1เทียบmanualmultisetfixtures ไม่ใช้LLMjudgeเป็นoracleของexactmetric
18. เปลี่ยนjudgew答案orderpolicyfixed→randomizedคงpairs/judge; reportwinrate/positionbiasพร้อมpairedCIและไม่ใช้ผลสุ่มseedเดียวสรุป
19. reproducibilityrecordเพิ่มperexamplemodeloutputs/normalizer/scorerhash/metricdefinition/CIseed-replicates/contaminationaudit/date ไม่ใช่เพียงaggregateaccuracy
20. ทดลอง“RAGเพิ่มThaiQAEM≥.05โดยunsupportedclaimsไม่เพิ่ม”; samebase/prompt/testset,pairedbootstrapCIของΔและblindcitationrubric
