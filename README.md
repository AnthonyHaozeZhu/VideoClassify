# 视屏分类
> 2022年8月写于南开大学

利用PyTorch和OpenCV等工具对视频进行简单的分类

主要目的是为了练习依稀视频的处理方法

数据集：HMDB51数据集（http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar）

### 现存问题
#### 9月8日
解决了训练中频繁出现NAN的问题，发现是有些视频抽取的帧读入后一部分直接读为NAN

目前有问题的数据有

``
./data/hmdb51_org_jpg/fencing/Die_Another_Day_-_Fencing_Scene_Part_1_[HD]_fencing_f_cm_np2_le_goo_0/0.jpg
./data/hmdb51_org_jpg/fencing/Die_Another_Day_-_Fencing_Scene_Part_1_[HD]_fencing_u_cm_np2_fr_goo_2/1.jpg
./data/hmdb51_org_jpg/fencing/Die_Another_Day_-_Fencing_Scene_Part_1_[HD]_fencing_u_cm_np2_fr_goo_2/2.jpg
./data/hmdb51_org_jpg/fencing/Die_Another_Day_-_Fencing_Scene_Part_1_[HD]_fencing_u_cm_np2_fr_goo_2/3.jpg
./data/hmdb51_org_jpg/fencing/Die_Another_Day_-_Fencing_Scene_Part_1_[HD]_fencing_u_cm_np2_fr_goo_2/4.jpg
./data/hmdb51_org_jpg/fencing/Die_Another_Day_-_Fencing_Scene_Part_1_[HD]_fencing_u_cm_np2_fr_goo_2/6.jpg
./data/hmdb51_org_jpg/fencing/Die_Another_Day_-_Fencing_Scene_Part_1_[HD]_fencing_u_cm_np2_fr_goo_2/7.jpg
./data/hmdb51_org_jpg/fencing/Die_Another_Day_-_Fencing_Scene_Part_1_[HD]_fencing_u_cm_np2_fr_goo_2/9.jpg
./data/hmdb51_org_jpg/fencing/Die_Another_Day_-_Fencing_Scene_Part_1_[HD]_fencing_u_cm_np2_fr_goo_2/10.jpg
./data/hmdb51_org_jpg/fencing/Die_Another_Day_-_Fencing_Scene_Part_1_[HD]_fencing_u_cm_np2_fr_goo_2/13.jpg
./data/hmdb51_org_jpg/fencing/Die_Another_Day_-_Fencing_Scene_Part_1_[HD]_fencing_u_cm_np2_fr_goo_2/14.jpg
./data/hmdb51_org_jpg/fencing/Die_Another_Day_-_Fencing_Scene_Part_1_[HD]_fencing_u_cm_np2_fr_goo_2/15.jpg
./data/hmdb51_org_jpg/brush_hair/Brushing_Her_Hair__[_NEW_AUDIO_]_UPDATED!!!!_brush_hair_h_cm_np1_fr_goo_0/1.jpg
./data/hmdb51_org_jpg/brush_hair/Brushing_Her_Hair__[_NEW_AUDIO_]_UPDATED!!!!_brush_hair_h_cm_np1_fr_goo_0/2.jpg
./data/hmdb51_org_jpg/brush_hair/Brushing_Her_Hair__[_NEW_AUDIO_]_UPDATED!!!!_brush_hair_h_cm_np1_fr_goo_0/5.jpg
./data/hmdb51_org_jpg/brush_hair/Brushing_Her_Hair__[_NEW_AUDIO_]_UPDATED!!!!_brush_hair_h_cm_np1_fr_goo_0/6.jpg
./data/hmdb51_org_jpg/brush_hair/Brushing_Her_Hair__[_NEW_AUDIO_]_UPDATED!!!!_brush_hair_h_cm_np1_fr_goo_0/7.jpg
./data/hmdb51_org_jpg/brush_hair/Brushing_Her_Hair__[_NEW_AUDIO_]_UPDATED!!!!_brush_hair_h_cm_np1_fr_goo_0/8.jpg
./data/hmdb51_org_jpg/brush_hair/Brushing_Her_Hair__[_NEW_AUDIO_]_UPDATED!!!!_brush_hair_h_cm_np1_fr_goo_0/9.jpg
./data/hmdb51_org_jpg/brush_hair/Brushing_Her_Hair__[_NEW_AUDIO_]_UPDATED!!!!_brush_hair_h_cm_np1_fr_goo_0/11.jpg
./data/hmdb51_org_jpg/brush_hair/Brushing_Her_Hair__[_NEW_AUDIO_]_UPDATED!!!!_brush_hair_h_cm_np1_fr_goo_0/12.jpg
./data/hmdb51_org_jpg/brush_hair/Brushing_Her_Hair__[_NEW_AUDIO_]_UPDATED!!!!_brush_hair_h_cm_np1_fr_goo_0/14.jpg
./data/hmdb51_org_jpg/brush_hair/Brushing_Her_Hair__[_NEW_AUDIO_]_UPDATED!!!!_brush_hair_h_cm_np1_fr_goo_0/15.jpg
./data/hmdb51_org_jpg/situp/Ab_Workout__(_6_pack_abs_)_[_ab_exercises_for_ripped_abs_]_situp_f_nm_np1_le_goo_2/1.jpg
./data/hmdb51_org_jpg/situp/Ab_Workout__(_6_pack_abs_)_[_ab_exercises_for_ripped_abs_]_situp_f_nm_np1_le_goo_2/2.jpg
./data/hmdb51_org_jpg/situp/Ab_Workout__(_6_pack_abs_)_[_ab_exercises_for_ripped_abs_]_situp_f_nm_np1_le_goo_2/3.jpg
./data/hmdb51_org_jpg/situp/Ab_Workout__(_6_pack_abs_)_[_ab_exercises_for_ripped_abs_]_situp_f_nm_np1_le_goo_2/4.jpg
./data/hmdb51_org_jpg/situp/Ab_Workout__(_6_pack_abs_)_[_ab_exercises_for_ripped_abs_]_situp_f_nm_np1_le_goo_2/8.jpg
./data/hmdb51_org_jpg/situp/Ab_Workout__(_6_pack_abs_)_[_ab_exercises_for_ripped_abs_]_situp_f_nm_np1_le_goo_2/9.jpg
./data/hmdb51_org_jpg/situp/Ab_Workout__(_6_pack_abs_)_[_ab_exercises_for_ripped_abs_]_situp_f_nm_np1_le_goo_2/10.jpg
./data/hmdb51_org_jpg/situp/Ab_Workout__(_6_pack_abs_)_[_ab_exercises_for_ripped_abs_]_situp_f_nm_np1_le_goo_2/11.jpg
./data/hmdb51_org_jpg/situp/Ab_Workout__(_6_pack_abs_)_[_ab_exercises_for_ripped_abs_]_situp_f_nm_np1_le_goo_2/12.jpg
./data/hmdb51_org_jpg/situp/Ab_Workout__(_6_pack_abs_)_[_ab_exercises_for_ripped_abs_]_situp_f_nm_np1_le_goo_2/13.jpg
./data/hmdb51_org_jpg/situp/Ab_Workout__(_6_pack_abs_)_[_ab_exercises_for_ripped_abs_]_situp_f_nm_np1_le_goo_2/14.jpg
./data/hmdb51_org_jpg/situp/Ab_Workout__(_6_pack_abs_)_[_ab_exercises_for_ripped_abs_]_situp_f_nm_np1_le_goo_1/1.jpg
./data/hmdb51_org_jpg/situp/Ab_Workout__(_6_pack_abs_)_[_ab_exercises_for_ripped_abs_]_situp_f_nm_np1_le_goo_1/2.jpg
./data/hmdb51_org_jpg/situp/Ab_Workout__(_6_pack_abs_)_[_ab_exercises_for_ripped_abs_]_situp_f_nm_np1_le_goo_1/3.jpg
./data/hmdb51_org_jpg/situp/Ab_Workout__(_6_pack_abs_)_[_ab_exercises_for_ripped_abs_]_situp_f_nm_np1_le_goo_1/4.jpg
./data/hmdb51_org_jpg/situp/Ab_Workout__(_6_pack_abs_)_[_ab_exercises_for_ripped_abs_]_situp_f_nm_np1_le_goo_1/5.jpg
./data/hmdb51_org_jpg/situp/Ab_Workout__(_6_pack_abs_)_[_ab_exercises_for_ripped_abs_]_situp_f_nm_np1_le_goo_1/6.jpg
./data/hmdb51_org_jpg/situp/Ab_Workout__(_6_pack_abs_)_[_ab_exercises_for_ripped_abs_]_situp_f_nm_np1_le_goo_1/7.jpg
./data/hmdb51_org_jpg/situp/Ab_Workout__(_6_pack_abs_)_[_ab_exercises_for_ripped_abs_]_situp_f_nm_np1_le_goo_1/11.jpg
./data/hmdb51_org_jpg/situp/Ab_Workout__(_6_pack_abs_)_[_ab_exercises_for_ripped_abs_]_situp_f_nm_np1_le_goo_1/12.jpg
./data/hmdb51_org_jpg/situp/Ab_Workout__(_6_pack_abs_)_[_ab_exercises_for_ripped_abs_]_situp_f_nm_np1_le_goo_1/13.jpg
./data/hmdb51_org_jpg/situp/Ab_Workout__(_6_pack_abs_)_[_ab_exercises_for_ripped_abs_]_situp_f_nm_np1_le_goo_1/15.jpg
./data/hmdb51_org_jpg/brush_hair/Brushing_Her_Hair__[_NEW_AUDIO_]_UPDATED!!!!_brush_hair_h_cm_np1_le_goo_2/6.jpg
./data/hmdb51_org_jpg/brush_hair/Brushing_Her_Hair__[_NEW_AUDIO_]_UPDATED!!!!_brush_hair_h_cm_np1_le_goo_2/9.jpg
./data/hmdb51_org_jpg/brush_hair/Brushing_Her_Hair__[_NEW_AUDIO_]_UPDATED!!!!_brush_hair_h_cm_np1_le_goo_2/11.jpg
./data/hmdb51_org_jpg/brush_hair/Brushing_Her_Hair__[_NEW_AUDIO_]_UPDATED!!!!_brush_hair_h_cm_np1_le_goo_2/12.jpg
./data/hmdb51_org_jpg/brush_hair/Brushing_Her_Hair__[_NEW_AUDIO_]_UPDATED!!!!_brush_hair_h_cm_np1_le_goo_2/14.jpg
./data/hmdb51_org_jpg/fencing/Die_Another_Day_-_Fencing_Scene_Part_1_[HD]_fencing_u_cm_np2_ba_goo_1/3.jpg
./data/hmdb51_org_jpg/fencing/Die_Another_Day_-_Fencing_Scene_Part_1_[HD]_fencing_u_cm_np2_ba_goo_1/4.jpg
./data/hmdb51_org_jpg/fencing/Die_Another_Day_-_Fencing_Scene_Part_1_[HD]_fencing_u_cm_np2_ba_goo_1/5.jpg
./data/hmdb51_org_jpg/fencing/Die_Another_Day_-_Fencing_Scene_Part_1_[HD]_fencing_u_cm_np2_ba_goo_1/6.jpg
./data/hmdb51_org_jpg/fencing/Die_Another_Day_-_Fencing_Scene_Part_1_[HD]_fencing_u_cm_np2_ba_goo_1/7.jpg
./data/hmdb51_org_jpg/fencing/Die_Another_Day_-_Fencing_Scene_Part_1_[HD]_fencing_u_cm_np2_ba_goo_1/8.jpg
./data/hmdb51_org_jpg/fencing/Die_Another_Day_-_Fencing_Scene_Part_1_[HD]_fencing_u_cm_np2_ba_goo_1/9.jpg
./data/hmdb51_org_jpg/fencing/Die_Another_Day_-_Fencing_Scene_Part_1_[HD]_fencing_u_cm_np2_ba_goo_1/10.jpg
./data/hmdb51_org_jpg/fencing/Die_Another_Day_-_Fencing_Scene_Part_1_[HD]_fencing_u_cm_np2_ba_goo_1/11.jpg
./data/hmdb51_org_jpg/fencing/Die_Another_Day_-_Fencing_Scene_Part_1_[HD]_fencing_u_cm_np2_ba_goo_1/12.jpg
./data/hmdb51_org_jpg/fencing/Die_Another_Day_-_Fencing_Scene_Part_1_[HD]_fencing_u_cm_np2_ba_goo_1/13.jpg
./data/hmdb51_org_jpg/fencing/Die_Another_Day_-_Fencing_Scene_Part_1_[HD]_fencing_u_cm_np2_ba_goo_1/14.jpg
./data/hmdb51_org_jpg/fencing/Die_Another_Day_-_Fencing_Scene_Part_1_[HD]_fencing_u_cm_np2_ba_goo_1/15.jpg
./data/hmdb51_org_jpg/fencing/Die_Another_Day_-_Fencing_Scene_Part_1_[HD]_avi_fencing_f_cm_np2_le_goo_0/1.jpg
./data/hmdb51_org_jpg/fencing/Die_Another_Day_-_Fencing_Scene_Part_1_[HD]_avi_fencing_f_cm_np2_le_goo_0/2.jpg
./data/hmdb51_org_jpg/fencing/Die_Another_Day_-_Fencing_Scene_Part_1_[HD]_avi_fencing_f_cm_np2_le_goo_0/4.jpg
./data/hmdb51_org_jpg/fencing/Die_Another_Day_-_Fencing_Scene_Part_1_[HD]_avi_fencing_f_cm_np2_le_goo_0/5.jpg
./data/hmdb51_org_jpg/fencing/Die_Another_Day_-_Fencing_Scene_Part_1_[HD]_avi_fencing_f_cm_np2_le_goo_0/6.jpg
./data/hmdb51_org_jpg/fencing/Die_Another_Day_-_Fencing_Scene_Part_1_[HD]_avi_fencing_f_cm_np2_le_goo_0/8.jpg
./data/hmdb51_org_jpg/fencing/Die_Another_Day_-_Fencing_Scene_Part_1_[HD]_avi_fencing_f_cm_np2_le_goo_0/9.jpg
./data/hmdb51_org_jpg/fencing/Die_Another_Day_-_Fencing_Scene_Part_1_[HD]_avi_fencing_f_cm_np2_le_goo_0/11.jpg
./data/hmdb51_org_jpg/fencing/Die_Another_Day_-_Fencing_Scene_Part_1_[HD]_avi_fencing_f_cm_np2_le_goo_0/12.jpg
./data/hmdb51_org_jpg/fencing/Die_Another_Day_-_Fencing_Scene_Part_1_[HD]_avi_fencing_f_cm_np2_le_goo_0/15.jpg
./data/hmdb51_org_jpg/fencing/Die_Another_Day_-_Fencing_Scene_Part_1_[HD]_fencing_u_cm_np2_fr_goo_3/5.jpg
./data/hmdb51_org_jpg/fencing/Die_Another_Day_-_Fencing_Scene_Part_1_[HD]_fencing_u_cm_np2_fr_goo_3/6.jpg
./data/hmdb51_org_jpg/fencing/Die_Another_Day_-_Fencing_Scene_Part_1_[HD]_fencing_u_cm_np2_fr_goo_3/7.jpg
./data/hmdb51_org_jpg/fencing/Die_Another_Day_-_Fencing_Scene_Part_1_[HD]_fencing_u_cm_np2_fr_goo_3/8.jpg
./data/hmdb51_org_jpg/fencing/Die_Another_Day_-_Fencing_Scene_Part_1_[HD]_fencing_u_cm_np2_fr_goo_3/9.jpg
./data/hmdb51_org_jpg/fencing/Die_Another_Day_-_Fencing_Scene_Part_1_[HD]_fencing_u_cm_np2_fr_goo_3/10.jpg
./data/hmdb51_org_jpg/fencing/Die_Another_Day_-_Fencing_Scene_Part_1_[HD]_fencing_u_cm_np2_fr_goo_3/11.jpg
./data/hmdb51_org_jpg/fencing/Die_Another_Day_-_Fencing_Scene_Part_1_[HD]_fencing_u_cm_np2_fr_goo_3/12.jpg
./data/hmdb51_org_jpg/fencing/Die_Another_Day_-_Fencing_Scene_Part_1_[HD]_fencing_u_cm_np2_fr_goo_3/13.jpg
./data/hmdb51_org_jpg/fencing/Die_Another_Day_-_Fencing_Scene_Part_1_[HD]_fencing_u_cm_np2_fr_goo_3/14.jpg
./data/hmdb51_org_jpg/fencing/Die_Another_Day_-_Fencing_Scene_Part_1_[HD]_fencing_u_cm_np2_fr_goo_3/15.jpg
./data/hmdb51_org_jpg/situp/Ab_Workout__(_6_pack_abs_)_[_ab_exercises_for_ripped_abs_]_situp_f_nm_np1_le_goo_0/2.jpg
./data/hmdb51_org_jpg/situp/Ab_Workout__(_6_pack_abs_)_[_ab_exercises_for_ripped_abs_]_situp_f_nm_np1_le_goo_0/3.jpg
./data/hmdb51_org_jpg/situp/Ab_Workout__(_6_pack_abs_)_[_ab_exercises_for_ripped_abs_]_situp_f_nm_np1_le_goo_0/4.jpg
./data/hmdb51_org_jpg/situp/Ab_Workout__(_6_pack_abs_)_[_ab_exercises_for_ripped_abs_]_situp_f_nm_np1_le_goo_0/5.jpg
./data/hmdb51_org_jpg/situp/Ab_Workout__(_6_pack_abs_)_[_ab_exercises_for_ripped_abs_]_situp_f_nm_np1_le_goo_0/7.jpg
./data/hmdb51_org_jpg/situp/Ab_Workout__(_6_pack_abs_)_[_ab_exercises_for_ripped_abs_]_situp_f_nm_np1_le_goo_0/8.jpg
./data/hmdb51_org_jpg/situp/Ab_Workout__(_6_pack_abs_)_[_ab_exercises_for_ripped_abs_]_situp_f_nm_np1_le_goo_0/10.jpg
./data/hmdb51_org_jpg/situp/Ab_Workout__(_6_pack_abs_)_[_ab_exercises_for_ripped_abs_]_situp_f_nm_np1_le_goo_0/11.jpg
./data/hmdb51_org_jpg/situp/Ab_Workout__(_6_pack_abs_)_[_ab_exercises_for_ripped_abs_]_situp_f_nm_np1_le_goo_0/13.jpg
./data/hmdb51_org_jpg/situp/Ab_Workout__(_6_pack_abs_)_[_ab_exercises_for_ripped_abs_]_situp_f_nm_np1_le_goo_0/14.jpg
./data/hmdb51_org_jpg/situp/Ab_Workout__(_6_pack_abs_)_[_ab_exercises_for_ripped_abs_]_situp_f_nm_np1_le_goo_0/15.jpg
./data/hmdb51_org_jpg/fencing/Die_Another_Day_-_Fencing_Scene_Part_1_[HD]_avi_fencing_u_cm_np2_fr_goo_2/2.jpg
./data/hmdb51_org_jpg/fencing/Die_Another_Day_-_Fencing_Scene_Part_1_[HD]_avi_fencing_u_cm_np2_fr_goo_2/3.jpg
./data/hmdb51_org_jpg/fencing/Die_Another_Day_-_Fencing_Scene_Part_1_[HD]_avi_fencing_u_cm_np2_fr_goo_2/5.jpg
./data/hmdb51_org_jpg/fencing/Die_Another_Day_-_Fencing_Scene_Part_1_[HD]_avi_fencing_u_cm_np2_fr_goo_2/6.jpg
./data/hmdb51_org_jpg/fencing/Die_Another_Day_-_Fencing_Scene_Part_1_[HD]_avi_fencing_u_cm_np2_fr_goo_2/7.jpg
./data/hmdb51_org_jpg/fencing/Die_Another_Day_-_Fencing_Scene_Part_1_[HD]_avi_fencing_u_cm_np2_fr_goo_2/9.jpg
./data/hmdb51_org_jpg/fencing/Die_Another_Day_-_Fencing_Scene_Part_1_[HD]_avi_fencing_u_cm_np2_fr_goo_2/10.jpg
./data/hmdb51_org_jpg/fencing/Die_Another_Day_-_Fencing_Scene_Part_1_[HD]_avi_fencing_u_cm_np2_fr_goo_2/11.jpg
./data/hmdb51_org_jpg/fencing/Die_Another_Day_-_Fencing_Scene_Part_1_[HD]_avi_fencing_u_cm_np2_fr_goo_2/12.jpg
./data/hmdb51_org_jpg/fencing/Die_Another_Day_-_Fencing_Scene_Part_1_[HD]_avi_fencing_u_cm_np2_fr_goo_2/13.jpg
./data/hmdb51_org_jpg/fencing/Die_Another_Day_-_Fencing_Scene_Part_1_[HD]_avi_fencing_u_cm_np2_fr_goo_2/14.jpg
./data/hmdb51_org_jpg/fencing/Die_Another_Day_-_Fencing_Scene_Part_1_[HD]_avi_fencing_u_cm_np2_fr_goo_2/15.jpg
./data/hmdb51_org_jpg/fencing/Die_Another_Day_-_Fencing_Scene_Part_1_[HD]_avi_fencing_u_cm_np2_ba_goo_1/3.jpg
./data/hmdb51_org_jpg/fencing/Die_Another_Day_-_Fencing_Scene_Part_1_[HD]_avi_fencing_u_cm_np2_ba_goo_1/4.jpg
./data/hmdb51_org_jpg/fencing/Die_Another_Day_-_Fencing_Scene_Part_1_[HD]_avi_fencing_u_cm_np2_ba_goo_1/5.jpg
./data/hmdb51_org_jpg/fencing/Die_Another_Day_-_Fencing_Scene_Part_1_[HD]_avi_fencing_u_cm_np2_ba_goo_1/6.jpg
./data/hmdb51_org_jpg/fencing/Die_Another_Day_-_Fencing_Scene_Part_1_[HD]_avi_fencing_u_cm_np2_ba_goo_1/7.jpg
./data/hmdb51_org_jpg/fencing/Die_Another_Day_-_Fencing_Scene_Part_1_[HD]_avi_fencing_u_cm_np2_ba_goo_1/8.jpg
./data/hmdb51_org_jpg/fencing/Die_Another_Day_-_Fencing_Scene_Part_1_[HD]_avi_fencing_u_cm_np2_ba_goo_1/9.jpg
./data/hmdb51_org_jpg/fencing/Die_Another_Day_-_Fencing_Scene_Part_1_[HD]_avi_fencing_u_cm_np2_ba_goo_1/11.jpg
./data/hmdb51_org_jpg/fencing/Die_Another_Day_-_Fencing_Scene_Part_1_[HD]_avi_fencing_u_cm_np2_ba_goo_1/12.jpg
./data/hmdb51_org_jpg/fencing/Die_Another_Day_-_Fencing_Scene_Part_1_[HD]_avi_fencing_u_cm_np2_ba_goo_1/13.jpg
./data/hmdb51_org_jpg/fencing/Die_Another_Day_-_Fencing_Scene_Part_1_[HD]_avi_fencing_u_cm_np2_ba_goo_1/14.jpg
/data/hmdb51_org_jpg/fencing/Die_Another_Day_-_Fencing_Scene_Part_1_[HD]_avi_fencing_u_cm_np2_fr_goo_3/2.jpg
./data/hmdb51_org_jpg/fencing/Die_Another_Day_-_Fencing_Scene_Part_1_[HD]_avi_fencing_u_cm_np2_fr_goo_3/3.jpg
./data/hmdb51_org_jpg/fencing/Die_Another_Day_-_Fencing_Scene_Part_1_[HD]_avi_fencing_u_cm_np2_fr_goo_3/6.jpg
./data/hmdb51_org_jpg/fencing/Die_Another_Day_-_Fencing_Scene_Part_1_[HD]_avi_fencing_u_cm_np2_fr_goo_3/7.jpg
./data/hmdb51_org_jpg/fencing/Die_Another_Day_-_Fencing_Scene_Part_1_[HD]_avi_fencing_u_cm_np2_fr_goo_3/9.jpg
./data/hmdb51_org_jpg/fencing/Die_Another_Day_-_Fencing_Scene_Part_1_[HD]_avi_fencing_u_cm_np2_fr_goo_3/10.jpg
./data/hmdb51_org_jpg/fencing/Die_Another_Day_-_Fencing_Scene_Part_1_[HD]_avi_fencing_u_cm_np2_fr_goo_3/11.jpg
./data/hmdb51_org_jpg/fencing/Die_Another_Day_-_Fencing_Scene_Part_1_[HD]_avi_fencing_u_cm_np2_fr_goo_3/12.jpg
./data/hmdb51_org_jpg/fencing/Die_Another_Day_-_Fencing_Scene_Part_1_[HD]_avi_fencing_u_cm_np2_fr_goo_3/13.jpg
./data/hmdb51_org_jpg/fencing/Die_Another_Day_-_Fencing_Scene_Part_1_[HD]_avi_fencing_u_cm_np2_fr_goo_3/14.jpg
Validating Epoch 0:   2%|▏         | 1/43 [00:01<01:20,  1.91s/it]./data/hmdb51_org_jpg/brush_hair/Brushing_Her_Hair__[_NEW_AUDIO_]_UPDATED!!!!_brush_hair_h_cm_np1_le_goo_3/1.jpg
./data/hmdb51_org_jpg/brush_hair/Brushing_Her_Hair__[_NEW_AUDIO_]_UPDATED!!!!_brush_hair_h_cm_np1_le_goo_3/2.jpg
./data/hmdb51_org_jpg/brush_hair/Brushing_Her_Hair__[_NEW_AUDIO_]_UPDATED!!!!_brush_hair_h_cm_np1_le_goo_3/3.jpg
./data/hmdb51_org_jpg/brush_hair/Brushing_Her_Hair__[_NEW_AUDIO_]_UPDATED!!!!_brush_hair_h_cm_np1_le_goo_3/4.jpg
/data/hmdb51_org_jpg/flic_flac/BHS___FlickFlack_[Tutorial]_flic_flac_f_cm_np1_le_med_0/2.jpg
./data/hmdb51_org_jpg/flic_flac/BHS___FlickFlack_[Tutorial]_flic_flac_f_cm_np1_le_med_0/3.jpg
./data/hmdb51_org_jpg/flic_flac/BHS___FlickFlack_[Tutorial]_flic_flac_f_cm_np1_le_med_0/4.jpg
./data/hmdb51_org_jpg/flic_flac/BHS___FlickFlack_[Tutorial]_flic_flac_f_cm_np1_le_med_0/5.jpg
./data/hmdb51_org_jpg/flic_flac/BHS___FlickFlack_[Tutorial]_flic_flac_f_cm_np1_le_med_0/6.jpg
./data/hmdb51_org_jpg/flic_flac/BHS___FlickFlack_[Tutorial]_flic_flac_f_cm_np1_le_med_0/7.jpg
./data/hmdb51_org_jpg/flic_flac/BHS___FlickFlack_[Tutorial]_flic_flac_f_cm_np1_le_med_0/8.jpg
./data/hmdb51_org_jpg/flic_flac/BHS___FlickFlack_[Tutorial]_flic_flac_f_cm_np1_le_med_0/9.jpg
./data/hmdb51_org_jpg/flic_flac/BHS___FlickFlack_[Tutorial]_flic_flac_f_cm_np1_le_med_0/10.jpg
./data/hmdb51_org_jpg/brush_hair/Brushing_Her_Hair__[_NEW_AUDIO_]_UPDATED!!!!_brush_hair_h_cm_np1_le_goo_1/2.jpg
./data/hmdb51_org_jpg/brush_hair/Brushing_Her_Hair__[_NEW_AUDIO_]_UPDATED!!!!_brush_hair_h_cm_np1_le_goo_1/3.jpg
./data/hmdb51_org_jpg/brush_hair/Brushing_Her_Hair__[_NEW_AUDIO_]_UPDATED!!!!_brush_hair_h_cm_np1_le_goo_1/4.jpg
``