function [ soundData, call, Fs ] = LoadData(baseDir, csvFile)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
    
    csvFile = strcat(baseDir, csvFile)

    %boolean array whether a certain file was from a right whale.
    call = csvread(csvFile, 1, 1); 
    
    soundDataList = dir(baseDir);
    soundData = zeros(1, size(baseDir));
    for i = 1:size(baseDir)
        soundData(i) = readwav(soundDataList.name);
    end
    [soundData(1), Fs] = readwav(soundDataList.name);
    
end

