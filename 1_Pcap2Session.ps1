# Get the exact current directory path
$baseDir = $PWD.Path

New-Item -ItemType Directory -Force -Path "$baseDir\2_Session\AllLayers"
New-Item -ItemType Directory -Force -Path "$baseDir\2_Session\L7"

foreach($f in gci "$baseDir\1_Pcap\*.pcap")
{
    # Pass absolute paths to SplitCap
    .\0_Tool\SplitCap_2-1\SplitCap.exe -p 50000 -b 50000 -r $f.FullName -o "$baseDir\2_Session\AllLayers\$($f.BaseName)-ALL"
    
    gci "$baseDir\2_Session\AllLayers\$($f.BaseName)-ALL" | ?{$_.Length -eq 0} | del

    .\0_Tool\SplitCap_2-1\SplitCap.exe -p 50000 -b 50000 -r $f.FullName -o "$baseDir\2_Session\L7\$($f.BaseName)-L7" -y L7
    
    gci "$baseDir\2_Session\L7\$($f.BaseName)-L7" | ?{$_.Length -eq 0} | del
}

.\0_Tool\finddupe.exe -del "$baseDir\2_Session\AllLayers"
.\0_Tool\finddupe.exe -del "$baseDir\2_Session\L7"