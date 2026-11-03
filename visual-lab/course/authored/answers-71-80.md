# เฉลยเฉพาะข้อ: advanced AI, research และ capstone
## 71
1. routerlogits[log.6,log.3,log.1],top2เลือกexperts0,1แล้วrenormalize[2/3,1/3]หากpolicytop-k-renorm; expert2ไม่activeสำหรับtokenนี้
2. assignments[[0,1],[0,2],[0,1]]ให้loads[3,2,1]; totalassignments6เท่ากับtokens3·topk2 ไม่เท่าจำนวนtokensเมื่อk>1
3. capacity=ceil(capacity_factor·N·k/E)ตามconventionนี้; N8,k2,E4,CF1→4,CF1.25→5; Switchtop1ใช้k1ต้องไม่สลับสูตร
4. loads[6,2,0,0],capacity3ทำoverflow3จาก8assignments=37.5%; ต้องระบุdrop/reroute/residualpolicyและtokenอาจยังมีexpertอีกตัว
5. Switch-styleaux=EΣfᵢPᵢ; E2,f[.75,.25],P[.6,.4]ให้2(.45+.1)=1.1; fเป็นfractionhardassign,Pmeanrouterprob
6. E8expertsตัวละ1M,shared.5M,total8.5M; top2activeประมาณ2.5Mต่อtokenแต่storageยัง8.5Mและroutingoverheadเพิ่ม
7. entropy=−Σp logp; uniform4ให้log4=1.386294,naturalunits; lowentropyไม่แปลbadเสมอหากspecializationเหมาะแต่ต้องดูloadbalance
8. jitterเฉพาะtraininglogits/inputตามpolicyด้วยseed; evalไม่มีjitterและrepeatable ไม่sampleexpertแบบtrainขณะbenchmarkdeterministic
9. experts8บนGPU4วาง2ตัว/GPU,tokenrouteไปownerแล้วกลับoriginalposition; sharedexpert/replicationต้องcountmemoryและbalancecommunication
10. trafficประมาณremoteassignments·d·bytesสำหรับdispatchและอีกเท่าตอนcombine; remote100,d1024,fp16ไปกลับ≈409600bytesก่อนmetadata/collectiveoverhead
11. top2weights.6/.3renormเป็น2/3,1/3; capacityoverflowต้องลดactualcontributionตามdroppolicyและไม่แสร้งทุกtokenได้2expertsครบ
12. topkIDsunique/inrange,weightsselectedsum1,loadsum=validassignments,capacitynotexceededและcombineกลับtokenorderถูก
13. CF1/1.25คงrouterlogits/tokens/seed; reportdroppedassignment/loadCV/activebytesและqualityหลังtrainingจริง ไม่อ้างdropลดคือqualityดีขึ้นแน่
14. recordrouterprobs(N,E),topIDs(N,k),loads(E),capacity,dispatchindices/expertoutputs/combinemasksและdrop-rate
15. เทียบtotalparamsMoEกับdenseโดยไม่reportactivecompute/storageหรือdroptokensทำperformanceclaimผิด; compute-matchedbaselineและcoverageชัด
16. routerO(NEd),expertcompute≈topk·expertFLOPs,alltoallตามremoteassignments; totalmemoryทุกexpertsไม่ลดตามk/Eโดยอัตโนมัติ
17. toyrouter/topkเทียบNumPyargsort/softmaxfixture; multideviceall-to-allต้องhardwareintegrationแยก ไม่ถือCPUloadarithmeticพิสูจน์GPUrouting
18. auxlossweight0→.01คงtrainingbudget/data/init; reportloadCV/drop/valNLLหลายseedsเพื่อดูbalance-qualitytradeoff
19. manifest มีexpert/sharedsizes/topk/renorm/capacityformula/drop/tiepolicy/auxloss/jitter/placement/dtype/tokenbatch/seed/hardware
20. ทดลอง“CF1.25ลดdrop≥50%โดยlatencyเพิ่ม≤10%”; CF1baseline,fixedskewedrequesttraceและmodelqualityonholdout
## 72
1. densecausalvalidpairs=N(N+1)/2; N4ได้10,N8ได้36; materializedscoretensorอาจยังN²entriesแม้maskครึ่งหนึ่ง
2. windowwรวมcurrentให้pairsΣmin(i+1,w); N8,w3ได้1+2+3·6=21เทียบdense36; w≥Nคืนdensecount
3. KVbytes=2LBnkvTdk·bytes,Tเพิ่ม2×cacheเพิ่ม2×; attentionfullprefillpairsเพิ่มใกล้4× ต้องแยกสองscalingนี้
4. MHAh8,GQAnkv2cacheลด4×เมื่อotherdimsfixed; actualscorequeryheadsยัง8และlong-contextqualityต้องประเมินใหม่
5. linearRoPEpositionsp'=p/s; p[0,4,8],s2→[0,2,4]; conceptualanglesเปลี่ยนแต่ไม่ได้รับประกันextrapolationถ้าmodelไม่train/adapt
6. docs8tokens,size4,overlap2ให้windows[0:4],[2:6],[4:8]; overlap0ให้2windows,overlap2ให้3และduplicatecoverageเพิ่ม
7. budget10,segmentslen[6,5,4]rankorderเลือก6,skip5,เลือก4รวม10ตามgreedyskippolicy; reservereply/prompttokensก่อนpacking
8. score=αcos+βrecency+γimportanceโดยแต่ละfeaturecalibrated; weights[.5,.3,.2],values[.8,.5,1]ได้.75; recencyหน่วยต้องnormalize
9. Recall@K=goldmemoriesretrieved/goldcountต่อquery; gold{a,b},top3[a,c,d]ได้.5; MRRวัดfirsthitต่างจากrecall
10. needlepositions0/25/50/75/100%×contextlengths,insertknownfactและaskunambiguousquery; scoreexactretrieval/factualanswerแยกและcontrolconflictingdistractors
11. N8,w3pairs21vsdense36ลด15pairsแต่implementationdense-maskยังcompute36/64ตามkernel; counttheoryไม่เท่ากับmeasuredspeed
12. paircountwindow≥N=dense,cachebyteslinearT,packing≤budgetและmemorytenant/expiryconstraintsไม่หลุดระหว่างrerank
13. window3/fullคงsequence/modelweights; reportneedleaccuracybydistanceและcomputedpaircounts/measuredlatencyแยก ไม่อ้างwindowเห็นทั้งอดีต
14. recordtokens/contextpositions/allowedpairmask/cachebytes/retrievedmemoryIDs/scores/packedlengthและposition-sweepaccuracy
15. วัดneedleเพียงต้นcontextหรือคำตอบเดาง่ายจากpretrainingทำlongcontextclaimเกินจริง; randomfacts/positionsและconflictingdistractorsholdout
16. slidingwindowtheoryO(Nwd)หากkernelทำsparseจริง,denseO(N²d); retrievalO(search+packing)และcacheยังขึ้นกับretainedtokens
17. paircountsเทียบexplicitbooleanmask.sumบนtinyN; cacheformulaเทียบtensorbytesและpackingเทียบmanualfixtureก่อนGPUbenchmark
18. recencyweight0→.3คงmemoryembeddings/queryset/importance; reportRecallกับstale-factrateหลายtemporalqueriesไม่ใช้latestเท่ากับcorrectเสมอ
19. manifest มีcontext/window/RoPEscaling/cachelayout/memoryembedding/timestamps/expiry/ACL/scoreweights/packing/needlepositions/seed
20. ทดลอง“retrievalmemoryลดstale-answer≥20%เทียบrecency-onlyโดยRecallลด≤.02”; fixedversionedfactsและtime-awareholdout
## 73
1. single-sampleaccuracyคือครั้งเดียวต่อโจทย์; self-consistencyโหวตcanonicalanswerจากหลายsamples ช่วยเมื่อerrorsไม่correlatedและcorrectanswerเป็นmodeแต่เพิ่มcompute
2. n10samples,c2correctให้pass@k=1−C(n−c,k)/C(n,k): pass1=.2,pass3=1−56/120=.533333,pass5=1−56/252=.777778; k≤n
3. budget1000tokens,N4ให้candidatecap250ก่อนverifieroverhead; realunusedtokensอาจreallocateตามpolicyที่กำหนด ไม่เรียกNเพิ่มว่าcomputeเท่าเดิม
4. arithmeticverifierparseexpressionด้วยASTallowlistแล้วevaluateexactFraction/integers เทียบanswer; rejectunknowncalls/hugeexponentsและไม่evalarbitrarycode
5. majorityใช้answerfrequency,bestofNใช้verifierscore; verifierผิดอาจเลือกwronganswerแม้majorityถูก จึงต้องindependentverifieraccuracy/calibration
6. solved_per_1k_tokens=1000·solved/totalgeneratedtokensรวมcandidatesที่ทิ้ง; solved20,tokens10000ได้2 ไม่ใช้เฉพาะselectedcompletiontokens
7. weakverifierให้คะแนนจากคำว่า“ถูกต้อง”ทำcandidatewrongแต่ใส่phraseชนะ; evalต้องsemantic/executabletestsและadversarialholdout
8. beamเก็บtopbpartialstatesทุกdepth,bestfirstขยายhighestprioritynode; toygraphS→A(cost1),B2,A→G5,B→G1optimalcost3ผ่านB ไม่greedycostedgeเดียว
9. outcomelabelตรวจfinalanswerส่วนprocesslabelตรวจstepvalidity; finalถูกจากเหตุผลผิดเป็นoutcomepassแต่processfail และต้องrubricกำหนดstepverification
10. freezequestions/model/prompt/totalgenerated+verifiertokenbudget/walltime; compareaccuracy/solvedper1k/latencyและdifficulty-slicesไม่best-of-Nกับsingleโดยไม่countcost
11. pass3fixture.533333ใช้unbiasedcombinatorialestimator ไม่ใช้1−(1−observedaccuracy)^3โดยอ้างexactestimateเดียวกัน
12. verifiercorrect/incorrect/invalidexpressionsfixture,passkbounded/monotonic,zero-correct0/allcorrect1และbudgetไม่เกินรวมcandidates
13. N1/4คงtotal1000tokensและverifierbudget; reporttruncation/accuracy/solvedper1kหลายseeds ไม่N4ใช้4×tokensเงียบๆ
14. recordcandidateIDs/answers/tokenlength/verifierscore/selectedID/groundtruth/processlabelsและtotalcompute ไม่ต้องเผยhiddenreasoningของผู้เรียน
15. tuningverifierบนbenchmarktestหรือignoringfailed/truncatedcandidatesทำpassดูดี; freezeheldoutและcountallattempts
16. Ncandidatecomputeประมาณsumtokens·decodecost+verification; beamอาจO(b·branching·depth)expandedstatesแต่cache/memoryเพิ่มตามliveprefixes
17. pass@kเทียบPythonmath.combexactintegers; arithmeticverifierเทียบhandcheckedFractionfixturesไม่ใช้LLMjudgeเป็นgroundtruthของarithmetic
18. majority→deterministicverifierselectionคงcandidatesเดียวกัน; reportselectionaccuracy/verifierfalseacceptและtokenbudget ไม่regenerateจนได้คำตอบที่ชอบ
19. manifest มีquestionhash/model/prompt/decodingseeds/N/tokenallocation/verifierversion/canonicalization/selectionrule/allcandidatecosts/metric
20. ทดลอง“verifiedbestof4เพิ่มaccuracy≥.05โดยsolvedper1kไม่ลด”; compute-matchedsinglebaselineและadversarialverifiercases
## 74
1. squareimage224,P14ได้16²=256patchtokens;448ได้32²=1024ยังไม่รวมCLS/separators; resolutionสองเท่าทำpatchtokens4×
2. visionX(B,N,768)@W(768,1024)+b→(B,N,1024); projectoroutputcompatibleLMwidthไม่เท่ากับvisionfeaturesมีภาษาalignmentแล้ว
3. budget2048,text500,images2×576,special20เหลือreply376; imageplaceholderอาจexpandmanytokensจึงต้องcountactualvisualtokensก่อนadmit
4. resize448singleP14ให้1024patches; tiles4×224ให้4×256=1024ก่อนper-tileCLS/globalthumbnail จึงtotalอาจมากกว่าที่คิด
5. normalizedxyxy[.1,.2,.5,.8],imageW100,H200→[10,40,50,160]; clip/validationsและletterboxinverseต้องตามpreprocessจริง
6. boxes[0,0,2,2]กับ[1,1,3,3]IoU1/7; coordinateconventioncontinuousและzeroarea-policyต้องตรงevaluation
7. scoreorderedmatchingGTsameclassIoU≥.5ไม่ซ้ำ; TP2,FP1,FN1ได้precision2/3,recall2/3 ไม่เอาคำบรรยายถูกแทนlocalizationcorrect
8. OCRsliceแยกfont/size/rotation/language/blur; measureCER/exactfieldmatchพร้อมgroundtruthlayoutและไม่ให้answerเดาจากdocumenttemplateอย่างเดียว
9. twoimagesA/Bถามattributeของภาพ1/2,swaporderพร้อมexpectedanswerสลับ; duplicate-imageและcontradicting-imagecasesจับidentityconfusion
10. auditclaimsทีละข้อพร้อมvisualevidence/region; fluencyไม่ใช่visualgrounding,includeunanswerable/occludedimageและrefusalpolicyที่เหมาะ
11. 448P14patches1024กับ224256,attentionpairsต่อvisualselfattentionใกล้16×ถ้าคงdense; downstreamLMcostตามtotaltokensรวมtext
12. projectorlastdimตรงLM,visualtokenbudgetไม่เกิน,boxconvertinverse-roundtripและimageswapidentitytestcorrect; imageabsentไม่แต่งvisualclaims
13. resizevs4tilesคงimage/questions/model/seed; reportOCRCER/smallobjectrecall/tokens/latencyไม่เลือกเฉพาะtileที่รู้คำตอบ
14. recordimageoriginal/resizeddimensions/patchIDs/projectedshape/contextslots/boxtransformsและgroundedaccuracyแยกOCR/multiimage
15. leakedcaptions/filenamesหรือtextquestionบอกanswerทำVLMดูเห็นภาพทั้งที่ไม่อ่าน; shuffled/no-imageablationsและgroupbyimageholdout
16. visualencoderattentionO(Nv²d),LMdenseattentionO((Nv+Nt)²d),projectorO(Nvdvdlm); tilingเพิ่มtokens/cache/latencyตามimplementation
17. projectionNumPyเทียบTorchcopiedweights,IoUmanualoracle; fullVLMqualityต้องdataset/learnerhumanrubricแยกไม่อ้างtoylinearpassยืนยันmultimodalunderstanding
18. linearprojector→MLPคงencoders/data/budget; reportgroundedaccuracy/latencyและparametercountหลายseed ไม่trainfullLMพร้อมกัน
19. manifest มีvision/LM/tokenizerhash/imagepreprocess/tiling/projector/visualspecialIDs/contextbudget/imageorder/boxconvention/evalslices/seed
20. ทดลอง“tilingเพิ่มsmallfontOCRexactmatch≥.1โดยvisualtokens≤2×”; resizebaseline,fixedimages/questionsและlatencybudget
## 75
1. 2วินาทีที่16k/24k/48kHzให้32000/48000/96000samplesต่อchannel; stereoเพิ่มcountstoredแต่ไม่duration
2. fs16000,frame20ms=320samples,hop10ms=160; ranges[0,320),[160,480),[320,640)และbufferเก็บoverlap160ระหว่างchunks
3. RMS=sqrt(mean(x²)); x[3,4]RMS√12.5≈3.535534,energymean12.5; VADRMS>thresholdและhysteresisตามpolicy ไม่ใช้sumenergyเมื่อframelengthเปลี่ยน
4. endpointsilence300msเพิ่มอย่างน้อย300msหลังlastspeechก่อนturnendบวกprocessing/lookahead; sweep200/500msวัดfalse-cutoffและlatencytradeoff
5. codecbitrate=frames_per_sec·codebooks·log2(codebooksize); 50fps,8books,1024entriesให้4000bits/sก่อนheaders/framing
6. audio tokens/sec=frames/sec·codebooksหากflattenallcodes; 50×8=400tokens/s ไม่เท่ากับspeechtexttokens/sec
7. RTF=processingseconds/audioduration; process.5s/audio2s→.25,แต่firstresponseอาจ.4s; lowRTFไม่รับประกันlowstreaminglatency
8. stateslistening→thinking→speaking; uservoiceในspeaking→cancelgeneration+flushqueuedaudio→listeningพร้อมgenerationIDกันlatechunksจากturnเก่า
9. buffer3packets,packet20msให้60msplayoutdelay; jitterbufferใหญ่ลดlate-dropแต่เพิ่มlatency,measurearrivalvariationจริง
10. slicesSNR/accent/device/overlap/connectionjitterพร้อมWER/CER/turnlatency/false-interruptและn; voicequalityhumanratingต้องแยกจากASRaccuracy
11. 50fps,8books,K1024ให้400tokens/sและ4000bits/s ต่างกัน10bits/token; bitrateไม่รวมtransport/encryptionheaders
12. streamingframerangesครบตามpolicy,cancelgenerationIDไม่รับstaleaudio,RTFpositive,VADsilenceไม่เป็นspeechและendpointfakeclockboundaries
13. endpoint200/500msคงrecordings/VAD/model; reportturnlatency/falsecuts/missedbargeinsและhumanconversationtrialnot_runหากยังไม่มี
14. recordwaveformchunks/samplepositions/VADstates/silencecounter/codeIDs/streamgenerationID/jitterqueueและTTFA/RTF/WER
15. วัดRTFofflineแล้วอ้างrealtimeconversationพร้อมbargeinไม่ได้; ต้องend-to-endstreamingtraceรวมnetwork/bufferingและunselectednoise/accentcases
16. waveformbufferO(sample_rate·window·channels),codecsequenceO(frames·books),runtimecomputeขึ้นกับmodel/chunk/contextและnetworkqueue
17. RMS/framecountเทียบNumPyfixture,CTCcollapsemanual,streamingstateเทียบfakeclockeventtrace; actualSTT/TTSqualityต้องseparateaudioeval
18. jitterbuffer20→60msคงpacketarrivaltrace; reportlate-loss/playoutlatency/underrunsหลายtracesไม่เปลี่ยนcodecพร้อมกัน
19. manifest มีsample-rate/frame/hop/VADthresholds/endpoint/lookahead/codec/codebooks/transport/jitterpolicy/cancelIDs/models/device/noiseevalhash
20. ทดลอง“barge-inยกเลิกaudioภายใน100msโดยfalseinterrupt≤2%”; nocancelbaselineในsimulationและrealconversationtrialแยกสถานะ
## 76
1. lineartransitionxnext=Ax+Bu; stackZ=[X,U],solveleast_squaresZΘ≈Xnextแล้วsplitΘ; xnext=.8x+.2u,tinyindependentinputsต้องrecoverA.8,B.2
2. one-stepใช้truecurrentstateทุกครั้ง ส่วนmulti-stepป้อนpredictionกลับ; errorอาจสะสม แม้one-stepMSEเล็ก ต้องploterrorbyhorizon
3. x0=0,xnext=x+u,goal2,actions[1,1]ถึง2cost0,[0,1]ถึง1terminalcost1; rolloutทุกcandidateจากsameinitialstate
4. J=Σ(x−goal)²+λΣu²; λสูงลดactionแรงแต่อาจถึงgoalช้า; λ≥0และunitsของstate/actionต้องnormalizeหรืออธิบาย
5. MPC: observecurrent→planhorizonH→executeactionแรก→observeใหม่→replan; ไม่executeopenloopทั้งsequenceแล้วเรียกว่าrecedinghorizon
6. pixelMSEเน้นtexture/backgroundแม้dynamicsสำคัญผิด; latentobjectiveลดdimensionแต่ต้องpreserveaction-relevantstateและevaluatecontroltask
7. ensemblepredictions[1,2,3]mean2,popvariance2/3เป็นuncertaintyproxy; sharedbiasทุกmodelอาจvarianceต่ำแต่ผิดทั้งหมด
8. biasedmodelxnext=x+2uแทนtruex+uทำplannerเลือกu1เพื่อgoal2แต่realถึง1; comparepredicted/actualreturnจับmodelexploitation
9. horizons1/2/4/8เริ่มsameheldoutstates/actions; recordMSE/uncertaintybyhorizonและไม่resetpredictionsด้วยtrueobsกลางopenloopโดยไม่บอก
10. evalรวมone/multistep,dynamicscalibration,controlsuccess/constraintviolations,distribution-shift/actionsunseen ไม่reconstructionอย่างเดียว
11. trueA.8,B.2,x1,u−1ให้xnext.6; rolloutu0ต่อได้.48; tableต้องแสดงทั้งpredicted/trueที่แต่ละt
12. knownlinearsystemrecoverytolerance,zeroactiondynamics,MPCexecuteoneactionthenreplanและuncertaintyvariance≥0; λnegativeinputreject
13. horizon2/8คงmodel/state/candidatebudget; reportactualgoalerror/compute/constraintviolationsไม่ใช้predictedreturnเป็นsolemetric
14. recordstates/actions/transitionpredictions/ensemblevariance/candidatecosts/chosenaction/actualnextstateและrollouterrorbyhorizon
15. fitmodelด้วยtesttrajectoryหรือevaluateonlytrainingactionsทำgeneralizationดูดี; splittrajectories/timeและOODaction/state-slices
16. planningcostO(candidates·horizon·model_forward),ensembleเพิ่มmodelsfactor; cachedlatentช่วยแต่mustmeasureและretainedtrajectoriesmemoryO(CHd)
17. least-squaresเทียบnp.linalg.lstsqfixtureและmanualrollout; simulatorcontrolperformanceแยกจากphysicalworldtransferซึ่งยังnot_run
18. uncertainty-penalty0→.1คงensemble/candidates/seed; reportactualreturn/model-exploitation/constraintviolationsหลายinitialstates
19. manifest มีsimulatorversion/state/actionunits/transitiondataset/split/modelensemble/plannerhorizon/candidategenerator/costλ/seed/constraints
20. ทดลอง“ensembleuncertaintypenaltyลดOODmodel-exploitation≥30%โดยnominalsuccessลด≤5จุด”; fixedtruesimulatorและholdoutstates
## 77
1. two-linkl1=l2=1: x=cosθ1+cos(θ1+θ2),y=sinθ1+sin(θ1+θ2); (0,0)→(2,0),(π/2,0)→(0,2),(0,π/2)→(1,1)
2. `u=np.clip(command,lower,upper)`elementwise; bounds[−1,1],command[−2,.5,3]→[−1,.5,1]; nonfinitecommandrejectและunitsต้องตรงactuator
3. proportionalu=K(target−x),xnext=x+dt·u; K1,dt.1,target1,x0ให้x1.1,x2.19; clipและstabilityต้องตรวจK·dt
4. cloningMSEmean((predexpert−actionexpert)²); predicted[1,3],expert[2,2]ได้1; heldouttrajectoriesไม่randomframesจากtrajectoryเดียว
5. actionchunksลดinferencefrequencyแต่reactช้าระหว่างchunk; chunk4ที่20Hzเปิดloop200msก่อนreplan ต่างone-action50ms
6. pathlengthΣ‖pnext−p‖,smoothnessเช่นΣ‖unext−u‖²; trajectory[(0,0),(1,0),(1,1)]length2ไม่ใช่straightdistance√2
7. 10/20/50Hzมีbudget100/50/20msต่อcycleรวมperception/planning/communication/actuation; worstcase/overrunต้องวัดไม่ดูmeanอย่างเดียว
8. measurementz=x+noise,estimateupdateเช่นEMA; reportRMSEกับgroundtruthและcontrollag ขณะที่noiseลดอาจlagเพิ่ม
9. sim-to-realslicesmass/friction/delay/noise/camera/lighting/actuatorlimitsในsimก่อน; ไม่deployhardwareจากtoytestและreportphysicalvalidationnot_run
10. VLAflowimages/proprioception/text→encoders/fusion→actionhead(chunks,dimension,units)→bounds/rate-limit→controller→simulatorfeedback; safetycontrollerอยู่นอกmodelproposal
11. l1=l2=1,angles0/π2ให้end(1,1); radiansไม่ใช่degreesต้องconvertก่อนtrig และjointframeconventionต้องsame
12. clippedactionswithinbounds/idempotentclip,FKknownposesตรง,P-controllererrorลดสำหรับstablegainและcycledeadlineviolationsนับจริง
13. chunk1/4คงsimulator/model/seed/targetchanges; reporttrackingerror/latency/action-smoothnessและresponseafterdisturbance
14. recordjointangles/radians/poses/observationnoise/stateestimate/proposed-clippedactions/clockdeadlineและtasksuccess/constraintviolations
15. frame-levelrandomsplitหรือexpertfutureactionเป็นinputทำBCMSEต่ำปลอม; splittrajectory/sessionและclosed-looprolloutevalเพื่อเห็นcompoundingerror
16. two-linkFKO(1),planningตามcandidates/horizon/modelcost; realtimebudgethardlimitไม่ใช่averageFLOPs และqueueingdelaysอาจdominant
17. NumPyFKเทียบanalyticknownposesและfinite-differenceJacobian; simulationcontrolleroracleไม่พิสูจน์realrobotstability/hardwaresafety
18. measurementfilteroff/EMA.2คงnoise/disturbancetrace; reporttrackingRMSE/lag/overshootโดยsamelooprateและactuatorbounds
19. manifest มีsimulator/frames/units/kinematics/dt/controllergains/actionbounds/chunklength/noiseseed/expertdata/split/latencyprotocol
20. ทดลอง“chunk4ลดmodelcalls75%โดยtrackingRMSEเพิ่ม≤10%ในslowtargets”; chunk1baseline,fixedtargets/disturbancesและdeadlinebudget
## 78
1. tensor1000elements rawFP32=4000bytes,FP16=2000,INT8=1000ก่อนquantmetadata/alignment; parametercountคงเดิมแม้bitsลด
2. peakRAM=residentweights+peakliveactivations+runtimearena+input/outputbuffers+stack; เช่น64+32+16+8+8=128KiB ไม่sumactivationsทุกlayerถ้าreusememoryได้
3. budget160KiB,estimated128KiB,safetymargin20%แบบmultiplyได้153.6KiBจึงfit; estimated140→168ไม่fit; margindefinitionต้องชัดไม่สลับdivide.8
4. graphoperators{Conv,ReLU,Resize}; backendAรองรับConv/ReLUจึงResizefallback,backendBรองรับครบ; dtype/layout/version constraintsต้องตรวจด้วยไม่เพียงopname
5. scale.1,zp0,x[−.25,0,.26]round-to-even→q[−2,0,3],dequant[−.2,0,.3]; error[.05,0,.04]และoverflowclipตามINT8range
6. distillationเปลี่ยนarchitectureลดparams,structuredpruningลบchannelsลดparams,quantizationลดbytesไม่จำเป็นลดparams; unstructuredmaskdenseอาจไม่ลดไฟล์
7. energy=P·time; .5W×.02s=.01J=10mJต่อinferenceก่อนidle/startup; วัดaverageenergyรวมdutycycleแยก
8. fs100Hz,window2s,channels3,float32buffer200·3·4=2400bytes; inferทุก.5swindowoverlap75%และdutycycle=compute_time/interval
9. ringcapacityNเก็บhead/size,writeindex=head%N,เมื่อเต็มoverwriteoldest; readchronologicalต้องwrapสองspansและconcurrencyproducer/consumerpolicyชัด
10. checklistoperatorcoverage/quantparams/layout/arenaRAM/latency/power/fallback/binaryversionและsensorcalibration; fallbackCPUอาจlatency/energyเกินแม้outputsถูก
11. resident64KiB+activation32+arena16+IO8+stack8=128KiB,20%reserve153.6; externalflashweightsยังมีtransfer/cachecostไม่เรียกว่าRAM0ทั้งหมด
12. memory-fitboundary/ringwrap/quantzero-map/opfallbackwarningsและCPUquantized-vs-floaterrortolerance; hardwaredeploymentยังแยกnot_run
13. INT8/FP32คงmodel/sensorwindows/preprocessing/device; reportquality/peakRAM/latency/energyจริง ไม่ใช้rawweightbytesแทนpeakRAM
14. recordwindowshape(T,C),bufferindices/quantcodes/scales/operatorbackend/live-tensorlifetimes/arena/highwatermarkและtaskaccuracy
15. measuredesktopkernelแล้วอ้างMCUlatencyหรือไม่รวมfallback/copiesทำclaimผิด; benchmarktargetdeviceและfullsensor-to-outputpipeline
16. ringwriteO(1)/sample,windowfeaturecostO(TC),modelcostตามoperators/kernel; peakarenaขึ้นกับtensorlivenessและalignmentไม่เพียงparameters
17. ringchronologicaloutputเทียบPythondeque(maxlen=N),quantfixtureNumPyและONNXCPUgraphauditแยกจากactualacceleratorruntime
18. dutyinterval.5→1sคงmodel/window/sensortrace; reportenergy/second,eventmissrateและend-to-endlatencyไม่onlyenergy/inference
19. manifest มีdevice/firmware/backend/graphops/layout/dtype/quantcalibration/window/rate/arena-plan/safety-margin/powermeasurement/seed/evaldata
20. ทดลอง“INT8fitRAM≤160KiBโดยaccuracyลด≤1จุดและp95≤20ms”; FP32baseline,targetdevicevalidationแยกจากCPUnumericalevidence
## 79
1. claimคือสิ่งที่ยืนยันเช่น“วิธีAลดerror”,evidenceคือผลวัด/ทดลองที่รองรับ,assumptionคือเงื่อนไขที่ใช้สรุปเช่นข้อมูลIID; assumptionไม่ใช่ผลพิสูจน์
2. ablationเปลี่ยนส่วนหนึ่งของระบบเพื่อวัดcontributionภายใต้ปัจจัยอื่นคงเดิม; ปิดmoduleAแล้วmetricลดอาจชี้ประโยชน์แต่ต้องคุมparameter/compute/confounders
3. seedกระทบinit/shuffle/dropout/sampling; seedเดียวไม่ยืนยันrobustnessและlibrary/hardwareเปลี่ยนอาจไม่bitwiseidentical ต้องrecordหลายแหล่งRNG
4. exactreproductionพยายามตรงdata/code/config/compute/protocolของงานเดิม; conceptreproductionทดสอบแนวคิดด้วยระบบขนาด/ข้อมูลต่างและไม่อ้างเลขbenchmarkเท่ากัน
5. fingerprintผูกconfig/code/data/seedเป็นdigestเพื่อบอกว่าสองrunsเหมือนอะไร; canonicalJSONsortkeysทำkeyorderไม่เปลี่ยนidentity แต่datarevisionเปลี่ยนต้องhashต่าง
6. baselineที่reproduceไม่ได้อาจมีpipeline/evaluationbug; proposalเหนือbaselineที่ผิดไม่พิสูจน์เหนือpaper ต้องปิดbaselinegapหรือจำกัดclaimให้ชัด
7. mean5seedsสะท้อนexpectedperformanceและvariance ส่วนbest-of5เลือกโชคดีทำselectionbias; reportทั้ง5ค่าmean/stdและbudgetครบ
8. original8GPUsแต่locallaptopหากลดmodel/data/budgetควรเรียกscaled/conceptreproductionพร้อมdifferences table ไม่exactreproductionของผลเต็ม
9. leakageทำmodelเห็นholdoutโดยตรงหรือทางpreprocessingทำmetricสูงแต่generalizationไม่จริง; conclusionจึงผิดแม้testsคำนวณถูก
10. proposalใช้tokens/epochs/searchbudgetมากกว่าทำimprovementมาจากcomputeได้; reportcompute-matchedและquality-vs-computecurveแทนเปรียบscoreอย่างเดียว
11. scores[.7,.8,.9,.8,.8]mean.8,sample_std≈.0707107,n5; `seed_statistics`ต้องระบุddofและCIassumptions ไม่เลือกbest.9เป็นexpected
12. relativeerrornearzeroใช้absdiff/max(eps,abs(reference))หรือsymmetricaldenominatorตามcontract; reference0,pred1e−12ควรตรวจabstoleranceไม่ตีเปอร์เซ็นต์อนันต์
13. canonicalserialize `json.dumps(x,sort_keys=True,separators=(',',':'))`; {'a':1,'b':2}กับreversekeyorderhashเท่ากัน แต่{'a':1,'b':3}ต่าง
14. pairedscoresA[.7,.8,.6],B[.8,.85,.65]ให้differences[.1,.05,.05],mean.0666667; pairbyseed/splitเดียวกันไม่sortscoresแยกแล้วจับคู่
15. resampleindices3ตัวร่วมA/Bด้วยreplacementBreplicatesแล้วCImeanΔpercentiles; identicalpairsΔ0ทุกreplicateและseedrepeatable; nเล็กCIยังไม่robust
16. ตรวจduplicatedIDs/near-duplicates,splitbeforefit,target/futurefeatures,checkpointselection,testnormalizationและmetriccodeกับpaperก่อนอ้างbreakthrough
17. trainlossmatchแต่valต่างอาจsplit/dedupต่าง,evalmode/dropoutผิด,preprocessingfitต่าง,tokenizer/lossmask/reductionต่างหรือcheckpoint/regularizationต่าง; ตรวจearliestdivergenceทีละfactor
18. baselineต่ำ15จุดแปลimplementation/protocol/computeยังไม่align; proposalgainอาจเพียงแก้baselinebug ต้องreportgapและไม่claimsuperiorityต่อpaper
19. contract1หน้าระบุclaim,exact-vs-scaledscope,model/data/protocoltable,baselineexpectedtolerance,compute/seedbudget,requiredtests,acceptancegatesและdeviations
20. เลือก3ablationsจากhypothesisเช่นnomoduleA,capacity-matchedreplacement,compute-matchedbaseline; fixedsplit/3seedsและallocatebudgetก่อนดูผล พร้อมprimarymetric/CI
## 80
1. tokenizerเรียนvocabulary/mergesจากtext; fitก่อนsplitทำเห็นtestdistributionและอาจleakrareforms จึงsplitdocumentก่อนแล้วfreezeTokenizerIDmapping
2. inputtokens[t]ทำนายtargettokens[t+1]; [1,2,3,4]→x[1,2,3],y[2,3,4]; ไม่targetเท่ากับinputและต้องshiftlossmaskด้วย
3. causalmaskปิดfuturekeysj>iเพื่อไม่ให้modelอ่านคำตอบtargetระหว่างtrain; paddingmaskอีกชุดปิดinvalidtokensและcacheoffsetต้องถูก
4. perplexity=exp(meanNLLovervalidtokens); trueprob.5ทุกtokenCElog2,PPL2; เทียบได้เมื่อtokenizer/evalcorpus/maskprotocolเหมือนกัน
5. KVcacheเก็บprojectedkeys/valuesทุกlayer/headของpasttokensเพื่อลดการคำนวณprefixซ้ำ; ไม่ใช่trainingdatasetmemoryและไม่จำเป็นเก็บQอดีต
6. trainingmemoryมีweights+gradients+optimizerstates/masterweights+savedactivations+temporarybuffers; parameterbytesอย่างเดียวประเมินRAM/VRAMต่ำเกิน
7. fullattentionมีquery-keypairs≈T²จึงcontext2×ทำpairs≈4×,KVcacheเพิ่ม2×; optimizedkernelsลดmaterializationไม่เปลี่ยนpaircountdense
8. pretrainingเรียนnexttokenจากcorpusกว้าง,SFTเรียนresponseตามinstruction/chattemplateมักmaskuserloss; weightsยังupdateทั้งคู่แต่data/objectiveconditioningต่าง
9. DPOreferenceให้baselinechosen/rejectedmarginเพื่อคุมpolicydeviation; referencefreezeและrevision/template/tokenizerต้องตรงcachedlogps
10. quantizedartifactอาจต้องdequantize/fallbackunsupportedkernel,batchเล็ก/IO/KVdominantจึงไฟล์เล็กแต่latencyไม่ลด; benchmarkactualruntimeและquality
11. เพิ่มtopkในgenerationหลังtemperature,keepklogitsแล้วsetothers−∞,sampleหรือgreedyตามflag; testk1เท่ากับargmaxและinvalidkraise ไม่แก้trainingloss
12. exportCSV/JSONแต่ละoptimizerupdateมีstep,tokens_seen,trainloss,valloss,lr,gradnormพร้อมmetricdefinition; plotfromexportและreloadตรวจpointsไม่หาย
13. checkpointmetadataJSONมีcode/data/tokenizer/modelconfighash,step/RNG/optimizer/scheduler/precision/seed/splitsและartifactdigests; validateก่อนresumeไม่เพียงjsonอ่านได้
14. lossmicrobatchหารaccumstepsเมื่อvalidcountsเท่ากัน,backwardKครั้งก่อนclip/step/schedule/zero_grad; variablepaddingใช้globalvalidtokennormalizationและtestเทียบfullbatch
15. configlayers/dim/headsต้องpositiveและdim%heads==0; architectureparametercounttestหลายconfig,KVshape/cachepositionถูกและloadcheckpointwrongconfigreject
16. เก็บlastvalidcheckpointและตรวจfirstnonfinitetensor: dataIDs/labels→logits→lossstable→grads→optimizerstate; ลดη/clipหลังหาสาเหตุและเพิ่มregressiontestไม่เปลี่ยนหลายอย่างพร้อมกัน
17. valต่ำผิดปกติตรวจdoc/near-dupข้ามsplit,tokenizerfitonval,futureshift/mask,paddingdenominatorและcheckpointtest-selection; comparehandcomputedtinyNLLก่อนเชื่อscore
18. Thaiไม่roundtripตรวจutf8bytesjoinก่อนdecode,normalizationirreversible,UNK/dropcharacters/specialIDcollisionและmergeorder; fixturesไทย/emoji/combining/emptyต้องกลับrawtextตามcontract
19. คำนวณPembedding≈Vd,blocks≈L(4d²+3ddffตามSwiGLU),headtyingpolicy; target1Mแล้วbudgetweights+grads+Adam+activationsก่อนเลือกL/d/h/context/batch; smokeCPUหนึ่งstepก่อนlargertraining
20. samecheckpoint/data/tokenizer/prompts/lengths/batch/device/warmup/seeds; qualityNLL+taskmetric,peakruntimebytesและTTFT/TPOT/p95แยก; INT8inferenceไม่เปรียบtrainingthroughputกับmixedprecisionโดยไม่ระบุ
