cl /O2 /LD /EHsc /MD %INCLUDE1% %INCLUDE2% mesh_inpaint_processor.cpp %LIB% ^
 /link /OUT:%OUTFILE% ^
 /LIBPATH:"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Tools\MSVC\14.29.30133\lib\x64" ^
 /LIBPATH:"C:\Program Files (x86)\Windows Kits\10\Lib\10.0.19041.0\um\x64" ^
 /LIBPATH:"C:\Program Files (x86)\Windows Kits\10\Lib\10.0.19041.0\ucrt\x64"