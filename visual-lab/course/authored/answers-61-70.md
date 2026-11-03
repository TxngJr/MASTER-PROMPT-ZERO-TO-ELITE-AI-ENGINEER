# เฉลยเฉพาะข้อ: production และ efficiency
## 61
1. symmetricINT8qmax127,scale=maxabs(x)/127; x=[−2,0,1,2]ได้scale2/127,q=[−127,0,64,127]เมื่อties-to-evenสำหรับ63.5; dequant=q·scale
2. asymmetricq0..255,scale=(xmax−xmin)/255,zp=round(−xmin/scale)แล้วclip; range[−1,3]ได้scale4/255,zp64และrepresentablezero=q64
3. tensor[[1,2],[100,200]]per-tensorใช้scale200/127ทำแถวแรกerrorสูงกว่าper-row scales[2/127,200/127]; reportMSEแยกchannelตามaxisที่ระบุ
4. groupedINT4แบ่งcontiguousweightsกลุ่มg,scale/zero-pointแต่ละกลุ่ม,quantize4bitsแล้วpackสองnibblesต่อbyte; paddinggroupท้ายต้องเก็บvalidlength
5. groupเล็กปรับdynamicrangeเฉพาะกลุ่มได้ละเอียดแต่scale-metadataเพิ่ม; g2/4/8คงW/codebookแล้วคำนวณMSE+totalbytes ไม่อ้างgเล็กดีที่สุดทุกmetric
6. clippingthresholdต่ำลดoutlierครองscaleแต่ทำoutliererrorมากขึ้น; วัดdownstream‖X(W−Wq)‖²บนcalibrationholdout ไม่ใช้weightMSEอย่างเดียว
7. 7BrawweightsFP16=14GB,INT8=7GB,INT4=3.5GBแบบdecimal ก่อนscales/zeros/alignment/KV/activations/runtimeworkspace
8. naiveRTNroundแต่ละweightโดยscale ส่วนGPTQใช้approximatecurvature/calibrationเพื่อชดเชยerrorของweightsที่quantizeแล้ว จึงขึ้นกับcalibrationdataและimplementation
9. AWQใช้activationstatisticsระบุsalientchannels/weightsและscaleก่อนquantizeเพื่อรักษาoutput ไม่ได้แปลว่าเก็บทุกoutlierเป็นFP16เสมอ
10. GGUFmetadataควรตรวจarchitecture/tensornames/shapes/quanttypes/blocksize/tokenizer/specialIDsและruntime-support; filename“Q4”อย่างเดียวไม่บอกทุกtensor4bits
11. x1,scale2/127ให้q64,dequant128/127≈1.007874,error≈.007874; ค่าx2แทนได้ตรงq127 แต่x1ไม่จำเป็นตรง
12. zero-mapถูก,qอยู่range,dequantshapeเดิม,packed-unpackedcodesตรงและscale>0; allzerochannelต้องpolicyscale1/q0ไม่หาร0
13. per-tensor/per-channelคงweights/calibration/codebook/bits; reportweightMSE/outputMSE/bytesและCPUruntimeหากมีkernelจริง
14. recordWshape/groupaxis/scales/zero-points/codes/dequant/inputactivationrangesและheldoutNLLหรือtaskmetric แยกquantizererrorจากservinglatency
15. calibrateด้วยtestหรือรายงานไฟล์เล็กเป็นinferenceเร็วโดยไม่benchmarkทำclaimผิด; calibrationtrainonlyและsamehardware/kernel/batch
16. quantizationconstructionO(P),packedstorage≈P·bits/8+metadata; matmulcostขึ้นกับnativekernel/fallback/dequantoverhead ไม่รับประกันspeedupตามbits
17. NumPyRTNเทียบexplicitround/clipfixtureและtorchquantizationเมื่อaxis/qrange/roundingตรง; GPTQ/AWQไม่ได้ใช้RTNoracleยืนยันอัลกอริทึมทั้งหมด
18. clipthresholdmax→percentile99คงW/calibration/seed; reportoutputerrorบนholdout/qualityหลายslicesไม่เลือกจากtestscore
19. manifest มีbasehash/calibrationIDs/quantalgorithm/bits/groupaxis/size/codebook/scales/rounding/kernel/runtime/dtype/qualitybenchmark/hardware
20. ทดลอง“per-channelINT8ลดoutputMSE≥30%เทียบper-tensorภายใต้metadataเพิ่ม≤5%”; fixedweights/calibrationและholdoutactivationfixture
## 62
1. KVbytes=2LBnkvTdk·bytes; MHAh8เทียบGQAnkv2ลด4×,MQAnkv1ลด8×เมื่อL/B/T/dk/dtypeเท่ากัน ไม่ลดqueryheadcomputeตามเดียวกัน
2. DynamicCacheโตตามactualcontextประหยัดจองล่วงหน้าแต่reallocation/compileยาก; StaticCacheจองmaxcontextช่วยfixedshapeแต่เสียreservedmemoryและต้องmaskunusedslots
3. lengths[2,4,6],padmax6ให้usefultokens12จาก18,efficiency2/3; bucketingลดpaddingแต่ต้องคุมqueuewait/latency
4. blocksize16,lengths[17,31]จอง32+32=64slotsใช้48,tailwaste16=25%; sharedprefix/allocatoroverheadต้องนับแยก
5. prefix1000tokensซ้ำใน10requests,cacheprefillreuseได้ประมาณ9000prefix-tokenprojectionsแทน10000ภายใต้identicalprefix/model/position ไม่ลดdecodeทั้งหมด
6. decodebatchเล็กอ่านweights/KVมากต่อtokenแต่FLOPsน้อยจึงbandwidth-boundได้; batchingเพิ่มreuseแต่เพิ่มlatencyและKVmemory
7. FlashAttentionจัดtile/fusionลดHBMtraffic/ไม่materializefullscores; denseattentionยังมีO(T²)paircompute ไม่เรียกlinearattention
8. acceptedtokens/totaldraftedtokens; drafted[4,4,4],accepted[3,1,4]rate8/12=2/3; end-to-endspeedรวมdraft/verifycostและrejecthandling
9. TTFTจากarrivalถึงfirsttoken,TPOTจากsubsequentintertokenหรือtotaldecode/(tokens−1)ตามdefinition; compareconcurrency1/4/8พร้อมp95และthroughput
10. continuousbatchschedulerทุกdecodeiterationปล่อยcompletedslots/รับwaitingเมื่อKVbudgetพอ; capqueue/maxbatch/maxtokensและrecordwaittime/admissionreason
11. lengths17/31block16waste16slots; เพิ่มblocksize32จอง64เท่าเดิมกรณีนี้ แต่distributionอื่นอาจต่าง จึงต้องวัดจากrequestlengthsจริง
12. cachelengthmonotonicwithinrequest,freedslotsreusable,no-cross-tenantprefixreuseโดยไม่authorizationและcached/full-logitsallclose
13. static/dynamicคงmodel/lengthdistribution/batch/dtype; reportreserved/allocatedpeak,TTFT/TPOT,warmupและcompilecostแยก
14. recordqueuearrival/admission/activeIDs/KVblocks/generatedlength/completion/TTFT/TPOTและtokens/sec; validstateต้องไม่overallocate
15. benchmarkเฉพาะdecodeหรือexcludequeuewaitทำservingดูเร็วเกิน; includeend-to-endSLOและreportprefill/cachehitdistribution
16. prefillattentionO(T²d),cacheddecodeO(Td)attentionต่อnewtoken; continuousbatchsimulatorcostขึ้นกับeventsแต่doesnotreplaceGPUbenchmark
17. KVformulaเทียบallocatedtensors.numel()*element_sizeบนCPUfixture; schedulerเทียบeventtraceด้วยconservationarrived=queued+active+done+dropped
18. prefixcacheoff/onคงrequests/tenantpolicy/model; reporthit-rate/TTFT/KVbytesและverifyoutputsidenticalหลายprompts
19. manifest มีmodel/tokenizer/cachelayout/blocksize/dtype/lengths/arrivalprocess/batchlimits/concurrency/prefixpolicy/hardware/runtime/warmup
20. ทดลอง“lengthbucketingเพิ่มpaddingefficiency≥20%โดยp95queuewait≤50ms”; FIFOconfigurationbaseline,fixedrequesttraceและsamecapacity
## 63
1. requestมีmodel:str,messages:[{role,content}],max_tokens:int≥1,temperature:float>0,stream:bool; validatesize/rolesและunknownfieldsตามAPIcontractversion
2. streamส่งchunkIDs/finish_reasonเรียงลำดับ; cancellationต้องหยุดgeneration/freeKVและปิดstream,ไม่เพียงตัดclientconnectionแต่workerยังใช้resource
3. livenessถามprocessยังตอบได้หรือไม่,readinessถามmodelloaded/warm/รับtrafficได้หรือไม่; loadingอาจlive=true,ready=false
4. sortedlatencies[10,20,30,40,50],linearquantilep50=30,p95=48,p99=49.6; วิธีquantileต้องตรงในdashboard/tests
5. LittleL=λWในstablelong-run; arrival20req/s,meanlatency.5sให้meaninflight10 ไม่ใช่p95concurrencyและใช้ไม่ได้แทนoverloadtransientตรงๆ
6. queuecapacity3,workers1; active1+queued3เต็มแล้วnextreject429/503ตามpolicyพร้อมretry-after; boundedqueueไม่สร้างservicecapacityเพิ่ม
7. tokenbucketrate100tokens/s,burst200; requestestimateprompt+max_new_tokensต้องreserveและreconcileactualusageอย่างatomicต่อtenant
8. replica≥ceil(λ·service_time/(workers_per_replica·target_utilization)); λ30/s,service.1s,1worker,target.6ให้5replicas
9. 5%canarytraffic2000requestsคาด100canary แต่randomactualอาจต่าง; rollbackหากp95/quality/errorเกินfrozengatesหลังminsamplewindow
10. manifestเก็บmodel/tokenizer/runtime/hardware/replicas/requesttrace/prompt-outputlength/concurrency/warmup/cachepolicy/metricsdefinitions/testdate
11. λ10/s,μworker4/s,workers2ให้capacity8/s,queueโตประมาณ2req/sก่อนcapacitylimit; timeoutเพิ่มไม่ทำμสูงขึ้น
12. readinessfalseก่อนload,trueหลังwarmup,cancelคืนslots,queueไม่เกินcapและinvalidschemaไม่ถึงworker; testsใช้fakeclock/workerไม่paidAPI
13. replicas2→4คงarrivaltrace/serviceassumptions; reportdrop/wait/p95utilizationจากsimulationและlabelว่าassumptions ไม่อ้างrealGPUcapacity
14. recordrequeststatesqueued/active/streaming/cancelled/done/rejected,remainingtokens/inflight/queueและp50/p95/p99พร้อมcount
15. loadtestclosedloopอย่างเดียวซ่อนoverloadหรือไม่รวมerrorsทำlatencyดูดี; reportoffered/accepted/completedrate,errorrateและarrivalmodel
16. queueeventsimO(eventslogevents)หากpriorityqueue; servingcomputeขึ้นกับtokens/model/hardware; memoryKVตามactivecontextไม่ใช่HTTPrequestcountอย่างเดียว
17. queuearithmeticเทียบmanualconservationtraceและNumPyquantilelinear; realloadgeneratorbenchmarkแยกจากsimulationพร้อมhardwareevidence
18. maxbatch1→4คงrequesttrace/runtimeและlatencybudget; reportthroughput/TTFT/TPOT/errorrateหลายrunsรวมqueuewait
19. reproducibilityเพิ่มarrivaltimestamps/payloadhash/timeout/retrypolicy/cancellation/quantilemethod/modeldigest/servercommitและloadgeneratorversion
20. ทดลอง“backpressureลดp99และOOMเป็น0ภายใต้overloadโดยsuccessrateตามSLO”; unboundedbaselineในsimulationและfixedtrace
## 64
1. imagebuildtaggitSHAและpinimagedigest; tagเดียวถูกoverwriteได้จึงdeploydigest@sha256 พร้อมsource/config/modelartifacthash
2. startupให้เวลาloadmodel,readinessกันtrafficก่อนwarm,livenessตรวจprocessdeadlockอย่างเบา; livenessไม่ควรล้มเพียงเพราะqueuebusy
3. desired4,maxSurge25%=1จึงpeak5pods,maxUnavailable25%=1จึงminimumavailable3; GPUcapacityต้องรองรับsurgeหรือrolloutติดpending
4. examplepodresourcesrequests/limits `nvidia.com/gpu: 1`; nodeต้องมีdeviceplugin/runtimeและavailableGPUจริง CPUvalidationYAMLไม่ยืนยันGPUdeploy
5. queue-depthmetric perpodtarget10,observedtotal50,current4ให้desiredceil(50/10)=5ตามsimplifieddesign; includeCPU/GPUloadingtimeและscale-downstabilization
6. ConfigMapเก็บnonsecretconfig,Secretเก็บcredentialแต่base64ไม่ใช่encryption; restrictRBAC/encrypt-at-restและอย่าบันทึกsecretsในrepo/log
7. Gatewayรับlistener/TLS,HTTPRouteจับhost/pathและชี้serviceweights; backendreadiness/timeout/bodylimitsต้องสอดคล้องstreamingLLM
8. 5%of2000=100expectedcanary,remaining1900champion; randomroutingต้องmeasureactualcounts/CIไม่บังคับexact100เว้นdeterministicassignment
9. gatesเช่นp95≤500ms,error≤1%,quality≥baseline−.02,minN100; failanyrequiredgateหยุดpromote/rollbackไปknowndigestพร้อมaudit
10. GPU8ตัว,1GPU/pod,usableheadroom2ตัวให้max6activepodsถ้าmemory/modelต่อGPUfit; multiGPU2/podให้3podsตามbudgetเดียว
11. desired4surge1ต้อง5GPUหากpodละ1และoldpodsยังอยู่; clusterมี4GPUrolloutอาจรอจึงต้องsurge0หรือเพิ่มcapacityตามavailabilitypolicy
12. manifestpinsdigests,probesแยกถูก,resourcebudgetไม่เกินcapacityและrollbackgatesdeterministic; noactualclusterdeployถือเป็นstaticvalidationเท่านั้น
13. surge0/1คงcapacity/startup/requestload; simulatorreportavailability/pendingdurationและSLO ไม่เรียกsimulationว่าพิสูจน์Kubernetesproductionแล้ว
14. recordpodstatespending/loading/ready/draining/terminated,replicacount/GPUs/queue/latency/qualitygateversion
15. ดูhealthHTTP200อย่างเดียวไม่ยืนยันmodelquality; canaryต้องsamepromptmixและenoughsamplesก่อนpromote ไม่ใช้metricaverageซ่อนtailfailure
16. deploymentcapacityคิดpods·GPU/requestedmemory,rollouttimeขึ้นกับload/drain/scheduling; controlplaneYAMLsizeไม่บอกmodelthroughput
17. validateYAMLschemaและsimulateprobe/rollouttransitionsด้วยfakeclock; actualGateway/GPUintegrationต้องแยกnot_runหากไม่มีcluster
18. readinessdelay0→warmup-completeคงrollout/load; reportcoldrequesterrors/TTFTและtime-to-readyไม่แก้modelพร้อมกัน
19. manifest มีimage/model/configdigests/cluster-runtime/deviceplugin/probes/resource/rollout/HPA/Gateway/trafficseed/gates/rollbacktarget
20. ทดลอง“warmupreadinessลดcoldstartp95≥30%โดยrolloutavailable≥3/4”; baselineearlyready,fixedstartuptraceในsimulationก่อนclustertrial
## 65
1. fingerprint=SHA256canonicalJSON(code,data_version,model,config,seed); data-v1/v2แม้rowscountเท่ากันต้องhashต่างและmanifestอ้างsourcechecksums
2. hashlib.sha256(artifactbytes); flipbyteหนึ่งตำแหน่งdigestเปลี่ยน; verifyก่อนloadและไม่ใช้filemtime/filenameแทนcontentintegrity
3. MLflowstart_run→log_params→log_metrics(step)→log_artifactพร้อมrunID; localfilebackendใช้ได้offline แต่connectiontestไม่แทนmodelquality
4. registrymodelversionผูกartifactdigest/runID/datasplit/metricversion/validationreport; ไม่registerชื่อversionอย่างเดียวโดยไม่มีprovenance
5. candidatealiasชี้versionรอgate,championชี้approvedversion; aliasเป็นpointerเปลี่ยนได้จึงlogก่อน/หลังพร้อมimmutableversionID
6. gatequality≥.8,p95≤200ms,requiredtestspass; score.82/latency210ต้องfail แม้qualityดี; missingmetricเป็นunknown/failไม่default0pass
7. PSIΣ(q−p)log(q/p); p[.5,.5],q[.25,.75]ได้≈.274653; bin0ต้องsmoothingpolicyและfixedtrainbinsไม่refitตามcurrent
8. missingrate=countnull/rows,freshness=now−latesteventtimeคนละsignalกับfeaturedistributiondrift; pipelineหยุดอาจdistributionเหมือนเดิมแต่stale
9. เก็บpredictionID/timestamp/modelversion/featureschema,joinlabelเมื่อมาถึงแล้วmetricตามcohortlabel-delay; อย่านับunlabeledเป็นnegativeทันที
10. rollbackrecordมีfrom/toimmutableversions,trigger/gatereport/time/actor/config/data-contractcompatibilityและverificationresultsหลังrollback
11. quality.82≥.8จริงแต่p95210>200ทำgatefail; ต้องแสดงเงื่อนไขที่ไม่ผ่าน ไม่เอาweightedscoreเฉลี่ยกลบrequiredgate
12. modifiedartifactdigestfails,canonicalfingerprintstable,keyorderไม่เปลี่ยนhash,promotionmissingmetricfailsและchampiontargetexists
13. thresholdquality.8/.85คงcandidate/evaldata; reportpromoterejectreasonsและfalsepromotionย้อนหลัง ไม่tunequalitygateจากincidenttestsetเดียว
14. recordrunID/modeldigest/datahash/per-slicequality/p95/missing/freshness/PSI/labelcoverageและpromotionstatetransition
15. เทียบmetricscorerversionต่างหรือdelayedlabelsไม่ครบทำclaimqualityผิด; freezecohort/censoringpolicyและreportcoverage/lagdistribution
16. fingerprintO(totalbytes)เมื่อhashartifact,PSIO(bins),streammonitorO(events); registrylatencyไม่ใช่inferenceperformanceและhistogrammemoryตามslices
17. PSIเทียบexplicitNumPysumfixtureและgateเทียบtruth-tableทุกrequiredcondition; MLflowoptionalintegrationseparateจากlocalmanifesttests
18. เพิ่มfreshnessgateคงmodel/requesttrace; reportstaleeventsblockedกับfalsealertsและdeploymentdelayไม่ใช้driftแทนfreshness
19. reproducibilityรวมrun/config/source/artifacthash,metricversion,evalsplit,gatepolicy,promotion/rollbackhistoryและtimestampทุกevidence
20. ทดลอง“freshnessgateตรวจpipelineหยุดภายใน5นาทีโดยfalsealert≤1/week”; baselinePSI-only,fixedhistoricalreplayและincidentlabels
## 66
1. contractevent_id:str,user_id:str,event_time:UTCtimestamp,ingested_at:UTCtimestamp,version:int≥1,region:enum,value:floatnullable; PK(event_id,version)และlatenesspolicyระบุ
2. เพิ่มoptionalnullablecolumncompatibleหากoldreadersignoreunknownตามcontract; rename/droprequired/type-narrowingเป็นbreakingต้องversion/migrateไม่silentcoerce
3. sort(event_id,versiondesc,ingested_atdesc)แล้วเลือกlatestด้วยdeterministictiepolicy; duplicateversionpayloadต่างต้องquarantineไม่สุ่มเลือก
4. partitiondate/regionช่วยprunequeriesแต่user_idcardinalityสูงทำfilesย่อยมาก; countpartitions≈days·regionsและobservefilesizesจริง
5. 10000filesเฉลี่ย10KBเป็นsmallfileproblem; compacttarget128MBโดยsamepartition/schemaและatomicmanifestcommit พร้อมrowcount/hashchecksก่อนลบoldversions
6. watermark=max_event_time−allowed_lateness; max12:00,lateness5minให้11:55,event11:54late,11:56ontime; processingtimeไม่แทนeventtime
7. keyhash(source,event_id,version,operation)ตามbusinessidentity; retryinputเดียวต้องsamekeyและdedupstateatomicไม่เพียงrandomUUIDต่อattempt
8. backfillpininputsnapshot/date-range/codeversion,เขียนnewpartitionversionแล้วatomicreplace,rerunให้logicalrowsเท่าเดิม; ไม่appendซ้ำทุกrun
9. lineagefingerprint=hash(sortedinputdigests+transformversion+config); datasetA/Borderสลับไม่เปลี่ยนเมื่อoperationcommutative แต่orderedjoinต้องบันทึกrolesด้วย
10. featureavailable_atต้อง≤prediction_time; event_timeเก่าแต่ingestedหลังpredictionยังเป็นfutureknowledge ต้องpoint-in-timejoinตามavailabilityจริง
11. latestversions[(a,1),(a,2),(b,1)]เหลือa2,b1; rerunappendinputเดิมผลเหมือนเดิมและeventcountsไม่เพิ่ม
12. idempotentrerun,latesttieconflictquarantine,schema-breakingreject,watermarkboundaryและpoint-in-timefuture-rowexcludedtests
13. lateness5/10minคงeventtrace; reportlate-dropfraction/state-retention/freshnessไม่เปลี่ยnsourceหรือdeduppolicyพร้อมกัน
14. recordschema/version/event/ingestion/availabilitytimes,partition/filecounts/rows,dedupkeysและfreshness/late-eventrate
15. joinfuturelabelหรือlate-arrivingfeatureโดยevent_timeอย่างเดียวทำMLqualityดีเกินจริง; usehistoricalavailabilitysnapshotและauditjoinexamples
16. hashdedupO(N)expectedtime/O(uniqueIDs)state,sortlatestO(NlogN),partitionpruningลดscanแต่too-manysmallfilesเพิ่มmetadataoverhead
17. latest-dedupเทียบPandasstable-sort/drop_duplicatesบนtinyfixture; Parquetschema/roundtripเทียบPyArrowCPUtestแยกจากdistributedstorageintegration
18. compactionoff/onคงlogicalsnapshot; assertrows/checksumssemanticตรงและmeasurefilecount/scantimeหลายrunsโดยcold/warmcachepolicyเหมือนกัน
19. manifest มีsourcesnapshots/schema/availabilitypolicy/dedup/tie/watermark/partition/compaction/backfillrange/codehash/outputchecksums/date
20. ทดลอง“point-in-timejoinลดoffline-onlinefeaturemismatchเป็น0”; eventtime-onlybaseline,fixedlateeventreplayและcoverage/freshnessbudget
## 67
1. timeout/503มักretryable,invalidschema/permissiondeniedpermanent,connectionresetหลังwriteunknownเพราะsideeffectอาจสำเร็จแล้ว; classify10casesพร้อมevidenceไม่retryทุกerror
2. base.1,max1,attempt0..4ให้caps[.1,.2,.4,.8,1]วินาที; fulljitterdrawUniform(0,cap)ด้วยseedและretrybudgetbound
3. at-least-onceworkerต้องatomicclaimidempotencykey+sideeffect/resultstate; messageซ้ำkeyเดิมคืนresultเดิมและไม่executewriteสองครั้ง
4. ρ=arrival/(workers·service_rate); arrival8,workers2,μ5ให้.8stableassumption,arrival12ให้1.2overload queueโตจนboundedreject
5. closedfailurethreshold2→open; หลังcooldown5s→halfopenอนุญาตprobeจำกัด; success→closed,failure→openและห้ามallrequestsprobeพร้อมกัน
6. modulo hash%Nเปลี่ยนNทำkeysย้ายมาก; rendezvousเลือกmaxhash(key,node)เพิ่มnodeย้ายเฉพาะkeysที่nodeใหม่ชนะ และremovednodekeysกระจายใหม่
7. quorumoverlapR+W>Nและwrite-write2W>Nภายใต้simplequorumassumptions; N3,R2,W2ผ่าน แต่overlapไม่ให้linearizabilityโดยลำพังเมื่อprotocolผิด
8. durations[1,1,1,1,10]มีoutlier10; thresholdเช่นmedian+3MADต้องguardMAD0; comparepercentiles/attemptageและstragglerไม่เท่ากับfailed
9. actorcheckpointมีactorID/version/mailboxcursor/processedkeys/statehash/code/schema/dependencyversions; recoveryต้องreplayหลังcursorโดยidempotent
10. controlplaneมีregistry/scheduler/config/health; dataplaneรับrequests/tokenstream/modelworkers; controlplanefailureไม่ควรทำexistingdataflowsล้มทุกครั้งหากcachedconfigvalid
11. arrival12/s,capacity10/sคิวเพิ่ม2/sก่อนreject; เพิ่มtimeoutไม่เปลี่ยนcapacityและอาจเพิ่มinflightmemory
12. duplicatewritesideeffectonce,circuitcooldown/halfopenbounded,queuecapacityinvariantและrendezvousunaffectedkeysไม่ย้ายเมื่อไม่ชนะnewnode
13. retry0/2คงfailuretrace/servicecapacity; reportsuccess/extraattempts/tail-latencyและretry-stormไม่ดูsuccessอย่างเดียว
14. recordattemptID/key/requeststate/circuitstate/queue/inflight/nodeassignment/recoverycursorและp95/drop/duplicatecount
15. benchmarkaverage-onlyซ่อนstragglersและไม่รวมfailedrequestsทำlatencyดูดีเกิน; reportofferedload/success/error/timeoutและtail
16. rendezvousO(nodes)ต่อkeynaive,moduloO(1),queueeventsimO(eventslogevents); retryเพิ่มworkตามattemptsและdedupstateต้องretentionpolicy
17. comparequeueconservationกับmanualtraceและrendezvoushashscoresfixed; distributednetworkpartitiontestsไม่ถูกแทนด้วยsinglenodeformulapass
18. jitteroff/onคงclients/retrycaps/seededfailurewindow; measurepeakconcurrentretry/totalattempts/successlatencyหลายseeds
19. manifest มีfailuretrace/backoff/jitter/circuit/idempotencyretention/hash/topology/quorum/timeout/capacity/codeversion/testclock
20. ทดลอง“fulljitterลดretryburstpeak≥50%โดยcompletionrateลด≤1จุด”; deterministicfailure-replaybaselinefixedbackoffและqueuebudget
## 68
1. trustedboundaries: userinput/toolresults/retrievedtextเป็นuntrusted,executorpolicy/identity/ACLtrusted; modelproposalข้ามไปtoolต้องผ่านvalidate+authorizeเสมอ
2. riskregisterแต่ละriskมีlikelihood1..5,impact1..5,scoreproduct,owner/control/test; exACLleakL3I5score15ไม่ใช่probability15%
3. authorize(identity,tool,args,resourceACL)เป็นdeterministicfunctionก่อนexecute; prompt“admin”ไม่เปลี่ยนidentityและmodelห้ามเลือกtenantcontextเอง
4. read-onlytoolsอนุญาตตามscope,writeขึ้นกับexplicittaskauthorizationและpolicy; storeapprovalcontextกับoperationargsเพื่อกันเปลี่ยนpayloadหลังอนุมัติ
5. filtertenant/groupก่อนchunksเข้าmodelและก่อนreranker/logging; testtenantAqueryไม่เห็นBtext/IDsแม้semanticmatchสูง
6. countersmax_prompttokens/max_generated/max_toolcalls/max_stepsแยก; reservebudgetก่อนexecuteและstopwithreasonเมื่อเกิน ไม่ตรวจหลังsideeffect
7. strictschemarequired/type/range/enum/additionalPropertiesFalse; path/resourceIDsต้องauthorizeเพิ่มแม้JSONvalid และไม่evalargumentsเป็นcode
8. balancebenign/violatingพร้อมsliceslanguage/attacktype; groundtruthpolicylabel/rationale/allowedresponseและfalse-positivecost ไม่ใช้adversarialอย่างเดียว
9. unsafeTP8,FN2,benignFP3,TN87ให้blockrecall.8,FPR3/90=.033333; refusalrateรวมไม่เท่ากับsafetyaccuracy
10. revoke/rotatecredential,disableaffectedtool,isolateaccess,retainredactedlogs,identifyexposure,remediate/retest,restorelimitedscopeพร้อมowner/timeline
11. riskL3I5=15prioritizationเท่านั้น; reducingL3→1หลังcontrolต้องมีadversarialtestevidenceไม่แก้ตัวเลขเพื่อให้dashboardเขียว
12. deniedcallsideeffect0,unknownfieldsreject,cross-tenantretrievalempty,budgetsboundedและbenigncasesไม่ถูกblockทั้งหมดเพื่อทำrecall1
13. prompt-onlyเทียบexecutorcheckคงmodel/attackset; reportunsafeexecutions/FPR/tasksuccessแยกและincludeobfuscatedtoolresults
14. recordidentity/authorizationcontext-redacted/schemaresult/policydecision/budget/tooltraceและper-slicerecall/FPR; no secretsinlogs
15. ใช้blockedrateสูงเป็นsuccessทำfalsepositiveหาย; ต้องbenignholdoutและblindpolicylabelพร้อมseparateexecutionpolicyfrommodelrefusal
16. ACLlookup/indexfiltercostตามresources/groups,budgetcheckO(1),schemaO(payloadsize); modelinferenceยังdominantแต่policycannotbeskippedforlatency
17. deterministicpolicyเทียบexplicitallow/denytruth-tableและmocktools sideeffectcounter; realincidentresponseintegrationต้องแยกจากtoytests
18. เพิ่มtool-outputinjectioncasesคงbenignset/model; comparepre/postexecutorpolicyโดยtraceและtasksuccessไม่ใช้modeltextrefusalอย่างเดียว
19. manifest มีpolicy/schema/ACLversion/identityclaims/registry/toolscopes/budgets/modelprompt/evalcasehash/incidentplaybook/testdate
20. ทดลอง“pre-retrievalACLทำunauthorizedchunkexposure0โดยauthorizedRecallลด≤.01”; postfilterbaseline,fixedmultitenantquerysetและtraceaudit
## 69
1. linearf(x)=2x1−x2+3 globalcoefficients[2,−1]บอกทุกจุด ส่วนlocalcontributionsเทียบbaseline0ที่x[1,2]คือ[2,−2]; intercept3แยก
2. importance=baselinescore−permutedscoreบนholdout; repeatshuffle5seedsreportmean/std; ใช้metricorientationถูกถ้าเป็นlossต้องสลับsign
3. x2≈x1และmodelใช้ทั้งคู่ shuffleหนึ่งอาจยังอีกตัวทดแทนจึงimportanceต่ำ; conditional/grouppermutationช่วยแต่ไม่ได้พิสูจน์causality
4. saliencyj=(f(x+h ej)−f(x−h ej))/(2h); f=2x1−x2ให้[2,−1]ทุกx,h1e−5,float64
5. IGj=(xj−baselinej)∫gradient_j(b+α(x−b))dα; linearcoeff[2,−1],x[1,2],baseline0ให้[2,−2],baseline[1,1]ให้[0,−1]
6. completenessgap=sumIG−(f(x)−f(b)); linearcaseต้อง≈0,nonlinearfinitequadratureมีerrorต้องreportgapไม่บังคับ0ด้วยแต่งค่า
7. gamev(S)=ΣweightsinS,w[1,2,3]ให้exactShapley[1,2,3]; interactionv({1,2})bonus2แบ่งinteraction1ให้ทั้งคู่ และsumφ=v(all)−v(empty)
8. f=x1*x2,x[2,3],baseline0ให้IG[3,3]แต่occludeทีละfeatureให้[6,6]; ต่างเพราะinteractionallocation ไม่ใช่หนึ่งimplementationผิดแน่
9. noiseεsmallfixedseedแล้วcompareattributioncos/rank/absolutechangeพร้อมpredictionchange; unstableexplanationอาจมาจากmodelboundaryต้องระบุ
10. randomizeparametersแล้วexplanationควรเปลี่ยนตามmodel; หากเหมือนเดิมตลอดอาจเป็นinputedgevisualizationมากกว่amodel-dependentattribution
11. f=x1*x2,baseline0,pathαxให้grad=[3α,2α],IG=[2·1.5,3·1]=[3,3],sum6ตรงf(x)−f(0)
12. linearIGexact/completenesswithin tolerance,Shapleyadditivity/efficiency,finite-diffgradientตรงanalyticและrandomized-modelsanitycheck
13. baseline0/meanคงmodel/input/integrationsteps; reportattributions/completenessและdomainmeaning ไม่เลือกbaselineที่ทำภาพดูตรงใจ
14. recordinput/baseline/pathpoints/gradients/integrationweights/attributionandgap,predictionsและstabilitymetric; attributionunitเป็นoutputdifference
15. explanationสวยหรือattentionweightsไม่พิสูจน์causalfeatureeffect; ใช้counterfactual/occlusion/holdoutพร้อมdomainconstraintsและavoidtesttuning
16. finite-diffO(d·forward),IGOm·backward,exactShapleyO(2^d·model); samplingapproxต้องCIและseed ไม่อ้างexactเมื่อdใหญ่
17. NumPyIGlinearเทียบCaptumIntegratedGradientsCPUด้วยsamebaseline/steps/method; completenessgapตามquadratureไม่ควรเทียบrawheatmapcolor
18. IGsteps16→128คงbaseline/model/input; reportgap/error/timeและไม่เปลี่ยนnormalizationเพื่อซ่อนผลต่าง
19. manifest มีmodelhash/inputschema/baseline/referencepopulation/method/steps/seed/featuregroups/metric/explanationrawvalues/date
20. ทดลอง“grouppermutationเผยimportancecorrelatedfeaturepairได้เสถียรกว่าindividual”; fixedsyntheticcorrelationและholdoutperformancedropหลายseeds
## 70
1. W[[1,0],[0,2]]มีzeros2/4sparsity.5,density.5; nearzeroไม่ถือzeroถ้าไม่ประกาศtoleranceและcountbiasแยกตามนิยาม
2. globalmagnitudepruneเลือกsmallestabsทั่วeligibleweights; [−.1,.2,−3,4],50%→[0,0,−3,4]tiepolicyต้องstable
3. layerwiseprune50%แต่ละlayerต่างglobal50%เมื่อscalesต่าง; globalอาจลบlayerหนึ่งเกือบหมด ต้องวัดqualityและperlayersparsity
4. rownormW[[.01,0],[3,4]]=[.01,5],ลบแถวแรกแล้วปรับnextlayerinputchannelตามmapping; zero-rowmaskเฉยๆยังไม่structuredshrink
5. sparsity0/.25/.5/.75คงsplit/budgetและreportqualityafterprune/fine-tuneแยก,actualparameter/serializedbytes/latency ไม่ใช้densityแทนspeed
6. densemaskedfloat32tensorยังเก็บทุกelement4bytesและmaskเพิ่มmemory; savingต้องsparseformatหรือremovechannels+supportedkernels
7. logits[0,log3],T1soft[.25,.75],T2≈[.366025,.633975],T4≈[.431765,.568235]; teacher/studentใช้Tเดียว
8. KD=T²KL(softmax(zteacher/T)||softmax(zstudent/T)); T²ชดเชยgradientscaleโดยประมาณและteacherstopgrad; hardCE+αKDต้องนิยามweights
9. sameinitialstudent/split/budget เปรียบhardCEกับhardCE+KD,frozenteacherและreportaccuracy/calibration/latencyหลายseeds
10. pruning+quantizationต้องbenchmarkcombinedartifact/runtimeเพราะsparsityอาจไม่ใช้kernelจริง; reportquality/filesize/peakmemory/p95เทียบoriginal
11. weights4elementsfloat32=16bytesแม้maskทำzeros50%; storedboolmaskอาจเพิ่ม4bytes,INT8raw4bytesก่อนscale ไม่เอาzero-countบอกsize
12. pruningmaskcountถูก,tiepolicyrepeatable,structuredchannelmapถูก,KD0เมื่อteacher/studentprobsเท่ากันและteachergradNone
13. global/layerwise50%คงW/trainbudget/seed; reportper-layerdensities/quality/actualruntime ไม่เปลี่ยnfinetuningstepsพร้อมmethod
14. recordWshape/mask/nonzero/channelmapping/teacher-studentlogits/T/hardloss/KDlossและquality-vs-latencycurve
15. teachertrainedontestหรือstudentfinetunedusingtestทำdistillationclaimผิด; sameevaluationdata/protocolและno-selectbestseed
16. pruningselectionO(PlogP)sortหรือO(P)selection; studenttrainingเพิ่มteacherforwardหากไม่cache,actualsparsecomputeขึ้นกับkernelไม่เพียงnonzeros
17. NumPysoftenedKLเทียบtorchKLdiv(logstudent,teacher,reduction='batchmean')·T²ด้วยaxis/reductionตรง; magnitudeprunefixtureexact
18. prune50%→75%คงfinetunebudget/seedและreportqualitydrop/latencymeasuredหลายrunsพร้อมstructured/unstructuredlabel
19. manifest มีbase/teacher/studenthash/pruningtargets/rate/ties/channelmap/quantconfig/T/α/data/splits/seed/runtimehardware/evalversion
20. ทดลอง“structuredchannelprune25%ลดp95latency≥15%โดยaccuracyลด≤1จุด”; densebaseline,samedevice/batch/runtimewarmupและ3seeds
