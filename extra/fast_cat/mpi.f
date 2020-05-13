      subroutine mpi_merge

      include 'genesis.def'
      include 'mpi.cmn'
      include 'input.cmn'
      include 'io.cmn'
      include 'field.cmn'
c
      integer i,ih,islice,mpi_tmp,status(MPI_STATUS_SIZE)
      character*255  cmd,opf
      character*10  nhs

      if (iphsty.le.0) return  ! no output at all
c
c     mpi requires some synchronization to guarantee that all files have been written
c
      if (mpi_id.gt.0) then
         call MPI_SEND(mpi_id,1,MPI_INTEGER,0,0,MPI_COMM_WORLD,mpi_err)
         return
      else
        do i=1,mpi_size-1
          call MPI_RECV(mpi_tmp,1,MPI_INTEGER,i,0,MPI_COMM_WORLD,
     c                     status,mpi_err)
        enddo
      endif
c
c     mergin main output files
c
!       do islice=firstout+1,nslice
!        if (mod(islice,ishsty).eq.0) then
!          write(cmd,10) outputfile(1:strlen(outputfile)),islice,
!      c                     outputfile(1:strlen(outputfile)) 
!          call system(cmd)
!          write(cmd,20) outputfile(1:strlen(outputfile)),islice
!          call system(cmd)
!        endif
!       enddo

      opf=trim(outputfile)

      cmd='cat '//trim(opf)//'.slice* >> '//trim(opf)
      print *, trim(cmd)
      call system(trim(cmd))
c
c     merging particle binary output
c
      if ((ippart.gt.0).and.(ispart.gt.0)) then
!         do islice=firstout+1,nslice
!           write(cmd,30) outputfile(1:strlen(outputfile)),islice,
!      c                     outputfile(1:strlen(outputfile)) 
!           call system(cmd)
!           write(cmd,40) outputfile(1:strlen(outputfile)),islice
!           call system(cmd)
!         enddo
! rm
        cmd='mv `ls '//trim(opf)//'.par.slice*|head -1`'
        cmd=trim(cmd)//' '//trim(opf)//'.par'
        print *, trim(cmd)
        call system(trim(cmd))
! cat
        cmd='cat '//trim(opf)//'.par.slice* >> '//trim(opf)
        cmd=trim(cmd)//'.par'
        print *, trim(cmd)
        call system(trim(cmd))
      endif
      if (idmppar.ne.0) then
!         do islice=firstout+1,nslice
!           write(cmd,31) outputfile(1:strlen(outputfile)),islice,
!      c                     outputfile(1:strlen(outputfile)) 
!           call system(cmd)
!           write(cmd,41) outputfile(1:strlen(outputfile)),islice
!           call system(cmd)
!         enddo
! rm
        cmd='mv `ls '//trim(opf)//'.dpa.slice*|head -1`'
        cmd=trim(cmd)//' '//trim(opf)//'.dpa'
        print *, trim(cmd)
        call system(trim(cmd))
! cat
        cmd='cat '//trim(opf)//'.dpa.slice* >> '
        cmd=trim(cmd)//trim(opf)//'.dpa'
        print *, trim(cmd)
        call system(trim(cmd))
      endif
c 
c
c     merging field binary output
c
      if ((ipradi.gt.0).and.(isradi.gt.0)) then
!         do islice=firstout+1,nslice
!           write(cmd,50) outputfile(1:strlen(outputfile)),islice,
!      c                     outputfile(1:strlen(outputfile)) 
!           call system(cmd)
!           write(cmd,60) outputfile(1:strlen(outputfile)),islice
!           call system(cmd)
!         enddo
! rm
        cmd='mv `ls '//trim(opf)//'.fld.slice* | head -1`'
        cmd=trim(cmd)//' '//trim(opf)//'.fld'
        print *, trim(cmd)
        call system(trim(cmd))
! cat
        cmd='cat '//trim(opf)//'.fld.slice* >> '
        cmd=trim(cmd)//trim(opf)//'.fld'
        print *, trim(cmd)
        call system(trim(cmd))
        do ih=2,nhloop
!           do islice=firstout+1,nslice
!            write(cmd,70) outputfile(1:strlen(outputfile)),hloop(ih)
!      c        ,islice,outputfile(1:strlen(outputfile)) ,hloop(ih)
!            call system(cmd)
!            write(cmd,80) outputfile(1:strlen(outputfile)),hloop(ih)
!      c                   ,islice
!            call system(cmd)
!          enddo
	  write(nhs,"(I1.1)") hloop(ih)
  ! rm
	  cmd='mv `ls '//trim(opf)//'.fld'//trim(nhs)//'.slice* |'
	  cmd=trim(cmd)//' head -1` '//trim(opf)//'.fld'//trim(nhs)
	  print *, trim(cmd)
	  call system(trim(cmd))
  ! cat
	  cmd='cat '//trim(opf)//'.fld'//trim(nhs)//'.slice*'
	  cmd=trim(cmd)//' >> '//trim(opf)//'.fld'//trim(nhs)
	  print *, trim(cmd)
	  call system(trim(cmd))
        enddo
      endif
      if (idmpfld.ne.0) then
!         do islice=firstout+1,nslice
!           write(cmd,51) outputfile(1:strlen(outputfile)),islice,
!      c                     outputfile(1:strlen(outputfile)) 
!           call system(cmd)
!           write(cmd,61) outputfile(1:strlen(outputfile)),islice
!           call system(cmd)
!         enddo
! rm
        cmd='mv `ls '//trim(opf)//'.dfl.slice* | head -1`'
        cmd=trim(cmd)//' '//trim(opf)//'.dfl'
        print *, trim(cmd)
        call system(trim(cmd))
! cat
        cmd='cat '//trim(opf)
        cmd=trim(cmd)//'.dfl.slice* >> '//trim(opf)//'.dfl'
        print *, trim(cmd)
        call system(trim(cmd))
        do ih=2,nhloop
!           do islice=firstout+1,nslice
!            write(cmd,71) outputfile(1:strlen(outputfile)),hloop(ih)
!      c        ,islice,outputfile(1:strlen(outputfile)) ,hloop(ih)
!            call system(cmd)
!            write(cmd,81) outputfile(1:strlen(outputfile)),hloop(ih)
!      c                   ,islice
!            call system(cmd)
!          enddo
	  write(nhs,"(I1.1)") hloop(ih)
! rm
          cmd='mv `ls '//trim(opf)//'.dfl'//trim(nhs)//'.slice*'
          cmd=trim(cmd)//' | head -1` '//trim(opf)//'.dfl'//trim(nhs)
	  print *, trim(cmd)
	  call system(trim(cmd))
! cat
	  cmd='cat '//trim(opf)//'.dfl'//trim(nhs)
	  cmd=trim(cmd)//'.slice* >> '//trim(opf)//'.dfl'//trim(nhs)
	  print *, trim(cmd)
	  call system(trim(cmd))
        enddo
      endif
c     
!       clean up
      cmd='rm '//trim(opf)//'*slice*'
      print *, trim(cmd)
      call system(trim(cmd))

 
      return
 10   format('less ',a,'.slice',I6.6,' >> ',a)
 20   format('rm ',a,'.slice',I6.6)
 30   format('cat ',a,'.par.slice',I6.6,' >> ',a,'.par')
 40   format('rm ',a,'.par.slice',I6.6)
 31   format('cat ',a,'.dpa.slice',I6.6,' >> ',a,'.dpa')
 41   format('rm ',a,'.dpa.slice',I6.6)
 50   format('cat ',a,'.fld.slice',I6.6,' >> ',a,'.fld')
 60   format('rm ',a,'.fld.slice',I6.6)
 70   format('cat ',a,'.fld',I1.1,'.slice',I6.6,' >> ',a,'.fld',I1.1)
 80   format('rm ',a,'.fld',I1.1,'.slice',I6.6)
 51   format('cat ',a,'.dfl.slice',I6.6,' >> ',a,'.dfl')
 61   format('rm ',a,'.dfl.slice',I6.6)
 71   format('cat ',a,'.dfl',I1.1,'.slice',I6.6,' >> ',a,'.dfl',I1.1)
 81   format('rm ',a,'.dfl',I1.1,'.slice',I6.6)

      return
      end


