# Future-long rows JSON 的只读内容寻址共用

此模块尚未接入任何训练生产者。它不处理当前20k、恢复02、已封存包或任何既有target目录，也不授予NN/环境/种子/优化器权限。

## 接口与边界

`new_stage(owner_sha256=...,stage_id=...)`只创建全新、独占的future-long storage staging路径。上下文给调用者`stage.payload`；未来生产者必须先在那里生成未最终绑定的目标文件，将八种`robot_i-{fit,calibration,eligibility,validation}-rows.json`候选设为0444，再调用`deduplicate_rows(stage,names=[...])`。每stage仅一次，错误或中断不自动重试。

只接受单链接正规文件；拒symlink、路径逃逸、跨文件系统、任意旧路径、来源/头部修改及已经存在result/completed/manifest等最终绑定的包。新helper只使用已有字节，不解析或重算NN数据。实际阶段必须由未来长预算的已登记owner在最终manifest/seal及优化器消费之前显式调用；当前native producer返回时已封存，因此不能直接在其返回后调用并声称已经集成。需要独立新版本producer/明确提前的存储回调契约。

CAS以SHA256命名，首次仅从该全新候选复制并逐字节重验；以后不重写、修补或改权限。相同SHA的新副本通过真实hardlink和原子替换共用inode。基准bytes/mode/mtime不变，目标逻辑路径/bytes/mode/SHA不变。inode共享会使目标mtime采用CAS基准mtime：只能在最终绑定前发生，并在收据显式记录；不声称全部POSIX metadata相同。

CAS的nlink/ctime因增加硬链接而变化是预期行为；不把这些当不可变科学字节。所有目标和CAS只读。任何写入都只发生在本模块创建的新namespace，历史源仅允许由调用者复制读取。

## 验证与空间口径

测试核对真实st_dev/st_ino/nlink、完整SHA、mode以及du分配空间。CAS初次复制有短暂额外空间，单个副本被移除的allocated bytes不等于整树净节约；净节约必须用同一根目录真实du前后差并考虑CAS和收据开销，不能把逐路径st_blocks求和当去重后的实际占用。

`new_probe(...)`只分配一个全新的临时目录，不能采用传入的旧目录。真实探针只可复制既有rows至该目录，原基准bytes/权限不能改；所有科学步骤、NN forward和新环境reset均为0。只有实际空间和恢复SHA验证通过，才能为新长预算的容量门提供工程证据。
