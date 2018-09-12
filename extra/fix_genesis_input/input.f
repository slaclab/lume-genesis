       subroutine initio
c     ==================================================================
c     manages all initial input for genesis
c     ------------------------------------------------------------------

      include 'genesis.def'
      include 'input.cmn'
      include 'io.cmn'
      include 'mpi.cmn'
      include 'field.cmn'

      integer ierr1,ierr2,i,ih
      character*34 file
      character*11  file_id
      character*4   file_ext
c
      nprobe=30                          !filenumber for filetype probing
      nlog=6
      call first
      ierr1=readin()                     !read namelist
c
      call touppercase(filetype)              !convert to upper case letter
      ftype=original
      i=index(filetype,'SDDS')
      if (i.gt.0) ftype=f_sdds
c
      if (ilog.ne.0) then
        if (index(outputfile,' ').eq.1) then
         file='log-file'
         goto 5
        else   
         file_id=''
         if (mpi_size.gt.1) then
           write(file_id,2) mpi_id         ! apppend process ID if run in MPI
         endif
 2       format('.node',I6.6)
         file=outputfile(1:strlen(outputfile))//'.log'//file_id         
         write(*,*) 'logfile: ',file
        endif   
        open(16,file=file,status='unknown',err=5)
        nlog=16
 5      if (nlog.eq.6)  i=printerr(erropen,file)
      endif
c
      ierr2=openoutputfile(ierr1,'template.in')        !open outputfile+dump namelist
      if ((ierr2+ierr1).lt.0) call last  !error occured?
c
      call chk_input                     !make input consistent
c
c     open addition output files
c
      if (ipradi.gt.0) then !is output of radiation field selected?
         irecfld=1
         nfld =openbinfile(outputfile,'fld '
     +                       ,10,8*ncar*ncar) !real and imag seprated
         do ih=2,nhloop 
           write(file_ext,100) 'fld',hloop(ih)
           nfldh(ih-1) =openbinfile(outputfile,file_ext,
     +                                 20+ih,8*ncar*ncar)	
	 enddo 
      else
         nfld=-1
      endif 
 100  format(A3,I1.1)
c
      if (ippart.gt.0) then  !is output of particle distribution desired?
         irecpar=1
         npar =openbinfile(outputfile,'par ',11,8*npart)
      else
         npar=-1
      endif

      if (idmpfld.ne.0) then   !dumped radiation field?
         ndump=openbinfile(outputfile,'dfl ',12,16*ncar*ncar)
	 do ih=2,nhloop
           write(file_ext,100) 'dfl',hloop(ih)
           ndumph(ih-1)=
     +         openbinfile(outputfile,file_ext,30+ih,16*ncar*ncar)
	 enddo
      else
         ndump=-1
      endif
c       
      if (idmppar.ne.0) then   !should the radiation field be dumped at the end?
         ndmp2=openbinfile(outputfile,'dpa ',13,8*npart)
      else
         ndmp2=-1
      endif
c
      if (itdp.ne.0) call opentimerec(ncar)          !prepare the crtime-record
c
c     open additional input files (maginfile is opened in magfield.f)
c
      call readbeamfile(beamfile)   !read external description file for beam 
      call readradfile(radfile)		!read external description file for radiation profile
c
      nfin=openbininput(fieldfile,14,ncar*ncar*16,1)! returns -1 if no name specified
      npin=openbininput(partfile,15,npart*8,0)      ! ---------- " ----------
      ndis=readdistfile(distfile,17)                ! ---------- " ----------
c
c     check for boundary violation or unphysical input parameters
c
      ierr1=chk_bnd()           !check for boundary violation
      if (ierr1.lt.0) call last
c
      return
c

      end  !of ioinit
c
      function openbininput(file,nio,size,isfield)
c     =================================================================
c     opens binary input files (filed and part files) and checks for 
c     the filetype
c     -----------------------------------------------------------------
c
      include 'genesis.def'
      include 'io.cmn'
c
      character*(*) file
      integer ft,nio,size,isfield

      openbininput=-1
      if (index(file,' ').eq.1) return   !no file selected

      openbininput=nio
      ft=detectfiletype(file)
      if (isfield.eq.1) then
         ftfield=ft           ! is field
      else 
         ftpart=ft            ! is particle
      endif
c
      open(nio,file=file,access='direct',status='old',
     +         recl=size,err=100) 
      return
c
 100  openbininput=printerr(erropen,file)
      call last
      return
      end
c      
c
      function detectfiletype(file)
c     =================================================================
c     the routine tries to read the beginning of the file
c     if the first line contains sdds then it returns the constant 
c     sdds, otherwise it returns the constant original
c     ----------------------------------------------------------------
c
      include 'genesis.def'
      include 'io.cmn'

      character*(*) file
      character*80 line
      integer i
c
      detectfiletype=original
      open(nprobe,file=file,status='old',err=100)
      read(nprobe,200,err=110) line
      close(nprobe)
      call touppercase(line)
      i=index(line,'SDDS')
      if (i.gt.0) detectfiletype=f_sdds
      return
c      
 100  detectfiletype=printerr(erropen,file) 
      call last     
      return
 110  detectfiletype=printerr(errread,file)
      call last      
      return
 200  format(a)
c
      end ! of detectfiletype
c
c
      function readin()
c     =================================================================
c     this routine reads in the user input files.
c     it assumes the standard fortran namelist format.
c     ------------------------------------------------------------------
c
      include 'genesis.def'
      include 'mpi.cmn'
      include 'input.cmn'
c
      integer nin,iarg,j,k
c
      namelist /newrun/
c
c     wiggler        
     +          aw0,xkx,xky,wcoefz,xlamd,fbess0,delaw,iertyp,iwityp,
     +          awd,iseed,awx,awy,
c     electron beam
     +          npart,gamma0,delgam,rxbeam,rybeam,alphax,alphay,
     +          emitx,emity,xbeam,ybeam,pxbeam,pybeam,cuttail,curpeak,
     +          conditx,condity,bunch,bunchphase,emod,emodphase,
c     radiation
     +          xlamds,prad0,zrayl,zwaist,pradh0,
c     grid-quantities   
     +          ncar,lbc,rmax0,dgrid,nscr,nscz,nptr,rmax0sc,iscrkup,
c     control
     +          nwig,delz,zsep,nsec,iorb,zstop,magin,magout,nbins,
     +          version,
c     strong focusing
     +          quadf,quadd,fl,dl,drl,f1st,qfdx,qfdy,sl,solen,
c     loading 
     +          ildgam,ildpsi,ildx,ildy,ildpx,ildpy,itgaus,lout,
     +          igamgaus,inverfc,
c     output
     +          iphsty,ishsty,ippart,ispart,ipradi,isradi,
     +          idump,iotail,nharm,iallharm,iharmsc,idmppar,
     +          idmpfld,ilog,ffspec,
c     external files
     +          beamfile,fieldfile,maginfile,magoutfile,outputfile, 
     +          partfile,distfile,filetype,radfile,
c     time-dependency
     +          curlen,ntail,nslice,shotnoise,iall,itdp,ipseed,isntyp,  
c     scan
     +          iscan,nscan,svar,scan,  
c     extension
     +          isravg,isrsig,eloss,ndcut,ibfield,imagl,idril,convharm,
     +          alignradf,offsetradf,multconv,igamref,
c     transfermatrix
     +          trama,itram11,itram12,itram13,itram14,itram15,itram16,
     +          itram21,itram22,itram23,itram24,itram25,itram26,
     +          itram31,itram32,itram33,itram34,itram35,itram36,
     +          itram41,itram42,itram43,itram44,itram45,itram46,
     +          itram51,itram52,itram53,itram54,itram55,itram56,
     +          itram61,itram62,itram63,itram64,itram65,itram66
c     temporary included parameter

c
c     initialize input/output
c
      nin=8
      readin=noerr
      call preset
      if (mpi_id.eq.0) then
        write(6,100)                                 !initialize input parameters
	  iarg = 1
	  inputfile  = ' '
	  call getarg (iarg, inputfile)
          j = index(inputfile//' ',' ')-1
          k = index(inputfile(1:j)//'.','.')-1
c        read(5,110) inputfile                        !get input filename.
      endif
      if (mpi_size.gt.1) then
        call MPI_BCAST(inputfile,30,MPI_CHARACTER,0,
     c                MPI_COMM_WORLD,mpi_err)
      endif
c
      open(nin,file=inputfile,err=10,status='old') !open input file.
      read(nin,newrun,err=20,end=20)                      !read in namelist
      close(nin)                                   !close file 
      return
c
 10   readin=printerr(erropenin,inputfile)
      return
 20   readin=printerr(errreadin,inputfile)
      call last
      return
c
c     format statements
c    
 100  format('Please enter input file name ')
 110  format(a30)
      end   !readin
c
      subroutine readbeamfile(file)
c     =============================================================
c     read the file for external description of the electron beam
c     -------------------------------------------------------------
c
      include 'genesis.def'
      include 'input.cmn'
      include 'time.cmn'
      include 'io.cmn'
c
      integer nin,i,ncol,ipar(15),ix,idata,j,ft,itmp
      character*(*) file
      character*50 cerr
      character*511 line
      real*8  ver,values(15),tmin,tmax,reverse
c
      itmp=0
      ndata=-1           
      if (index(file,' ').eq.1) return
c
      if ((iscan.gt.0).and.(iscan.lt.23)) then
          i=printerr(errscanex,file)
          return
      endif    
c
c     read file
c
      ft=detectfiletype(file)        ! check for filetype
c
      nin=opentextfile(file,'old',8)
      if (nin.lt.0) call last ! stop program on error
c
      ndata=-1            !# of rows not defined
      idata=0             !# of rows read
      ncol =15            !# of elements per line
      reverse=1.0         !tail for ZPOS and head for TPOS comes first in file
      ver=0.1
      do i=1,ncol
         ipar(i)=i        !basic order of input
      enddo
c
 1    read(nin,100,err=20,end=50) line
c
c     processing line
c     
      call touppercase(line)
      call getfirstchar(line,ix)
c
      if (ix.eq.0) goto 1                     !empty line
      if (line(ix:ix).eq.'#') goto 1          !no comment used
      if (line(ix:ix).eq.'?') then
         call getbeamfileinfo(line,ipar,ncol,ndata,ver,reverse)  !check for add. info
         goto 1
      endif
c
      if ((ndata.lt.0).and.(ver.lt.1.0)) then !old version
         i=extractval(line,values,1)
         if (i.lt.0) then
            write(cerr,*) 'Line number of BEAMFILE cannot be determined'
            i=printerr(errinput,cerr)
         endif
         ndata=nint(values(1))
         goto 1
      endif
c
      i=extractval(line,values,ncol)
      if (i.lt.0) then
        write(cerr,*) 'BEAMFILE data line ',idata+1,' has bad format'
        i=printerr(errinput,cerr)
        call last
      endif
c
      idata=idata+1
c
c     set default values      
c
      tpos(idata)=xlamds*zsep*idata    !can we make this the default?
      tgam0(idata)=gamma0
      tdgam(idata)=delgam
      temitx(idata)=emitx
      temity(idata)=emity
      txrms(idata)=rxbeam
      tyrms(idata)=rybeam 
      txpos(idata)=xbeam
      typos(idata)=ybeam
      tpxpos(idata)=pxbeam
      tpypos(idata)=pybeam
      talphx(idata)=alphax
      talphy(idata)=alphay
      tcurrent(idata)=curpeak
      tloss(idata)=eloss
c
c     write over with input data
c
      do j=1,ncol
         if (ipar(j).eq.1 ) tpos(idata)    = reverse*values(j)
         if (ipar(j).eq.-1) tpos(idata)    =-reverse*values(j)*3e8  ! time input
         if (ipar(j).eq.2 ) tgam0(idata)   = values(j)
         if (ipar(j).eq.3 ) tdgam(idata)   = values(j)
         if (ipar(j).eq.4 ) temitx(idata)  = values(j)
         if (ipar(j).eq.5 ) temity(idata)  = values(j)
         if (abs(ipar(j)).eq.6 ) txrms(idata)   = values(j) ! save for dual input
         if (abs(ipar(j)).eq.7 ) tyrms(idata)   = values(j) ! RXBEAM/BETAX
         if (ipar(j).eq.8 ) txpos(idata)   = values(j)
         if (ipar(j).eq.9 ) typos(idata)   = values(j)
         if (abs(ipar(j)).eq.10) tpxpos(idata)  = values(j) ! save for dual input
         if (abs(ipar(j)).eq.11) tpypos(idata)  = values(j) ! PXBEAM/XPRIME
         if (ipar(j).eq.12) talphx(idata)  = values(j)
         if (ipar(j).eq.13) talphy(idata)  = values(j)
         if (ipar(j).eq.14) tcurrent(idata)= values(j)
         if (ipar(j).eq.15) tloss(idata)   = values(j)
      enddo 
c
c     check for unphysical parameters
c
      if ((tgam0(idata)-4*abs(tdgam(idata))).lt.1) then
         itmp=printerr(errinput,'Energy GAMMA0 too small in BEAMFILE')  !abort
      else
        do j=1,ncol           ! calculate beam sizes (avods floating point exception
          if (ipar(j).eq.-6 )
     c     txrms(idata)=sqrt(txrms(idata)*temitx(idata)/tgam0(idata))
          if (ipar(j).eq.-7 )
     c     tyrms(idata)=sqrt(tyrms(idata)*temity(idata)/tgam0(idata))
           if (ipar(j).eq.-10 )
     c     tpxpos(idata)=tpxpos(idata)*tgam0(idata)
          if (ipar(j).eq.-11 )
     c     tpypos(idata)=tpypos(idata)*tgam0(idata)
        enddo
      endif  
c
      if (tcurrent(idata).lt.0) then
         itmp=printerr(errinput,'Current negative in BEAMFILE')  !abort
      endif 
c
      if (temitx(idata).lt.0) then
         itmp=printerr(errinput,'EMITX negative in BEAMFILE')  !abort
      endif 
c
      if (temity(idata).lt.0) then
         itmp=printerr(errinput,'EMITY negative in BEAMFILE')  !abort
      endif 
c
      if (idata.eq.1) then
         tmin=tpos(idata)
         tmax=tpos(idata)
      else
         if (tpos(idata).gt.tmax) tmax=tpos(idata)
         if (tpos(idata).lt.tmin) tmin=tpos(idata)
      endif
c
      goto 1
c
 50   close(nin)
c 
      if ((ndata.ge.0).and.(idata.ne.ndata)) then
         i=printerr(errinwarn,'BEAMFILE has fewer lines than defined')
      endif
      ndata=idata
      if (idata.lt.2) then
       i=printerr(errinput,'BEAMFILE contains less than 2 valid lines')
       call last
      endif
      if (itmp.ne.0) then
       call last
      endif
c
      if (ver.ge.1.0) then
        do i=1,ndata
          tpos(i)=tpos(i)-tmin   !set time window to zero
        enddo
      endif
c
      if (nslice.le.0) then
         nslice=int((tmax-tmin)/xlamds/zsep)
         if (ver.ge.1.0) then
           ntail=0
         else
           ntail=int(tmin/xlamds/zsep)
         endif
         write(ilog,110) nslice,ntail
      endif
c
      return
c
 20   i=printerr(errread,file)
      close(nin)
      call last
      return
c
c     format statement
c
 100  format(a)
 110  format('Auto-adjustment of time window:',/,
     c       'nslice=',i6,/ 
     c       'ntail =',i6)
c
      end     !readbeamfile
c 
      subroutine getbeamfileinfo(line,ipar,ncol,ndata,ver,reverse)
c     =================================================================
c     extract information from beamfile
c     -----------------------------------------------------------------
c
      include 'genesis.def'
c
      character*(*) line
      character*511 cline
      integer ndata,ipar(*),i,n,ncol,ierr,j,ix1,ix2,haszpos,iarg
      real*8  ver,val,reverse
c
c     version number
c
      i=index(line,'VERSION') !check for version number
      n=len(line)
      if (i.gt.0) then
        ierr=extractnumber(line(i+7:n),val)
        if (ierr.lt.0) then
           i=printerr(errinwarn,
     c               'Unrecognized information line in BEAMFILE')
        else
           ver=val
        endif
        return
      endif 
c
c     reverse order
c
      i=index(line,'REVERSE')
      n=len(line)
      if (i.gt.0) then
         reverse=-1.
         return
      endif
c         
c     line numbers
c
      i=index(line,'SIZE') !check for size argument (aka ndata)
      if (i.gt.0) then
        ierr=extractnumber(line(i+4:n),val)
        if (ierr.lt.0) then
           i=printerr(errinwarn,
     c               'Unrecognized information line in BEAMFILE')
        else
           ndata=nint(val)
        endif
        return
      endif
c
c     colum order
c
      i=index(line,'COLUMNS') !check for colums headers
      if (i.gt.0) then
         do j=1,15
            ipar(j)=0
         enddo
         ncol=0
         cline=line(i+7:n)
         haszpos=0
 1       call getfirstchar(cline,ix1)
         if (ix1.gt.0) then
           ix2=255             !search backwards
           do j=255,ix1+1,-1   !for first space after ix1
             if (cline(j:j).eq.' ') ix2=j  
           enddo
           iarg=0
           if (index(cline(ix1:ix2),'ZPOS').ne.0) iarg=1
           if (index(cline(ix1:ix2),'TPOS').ne.0) iarg=-1
           if (index(cline(ix1:ix2),'GAMMA0').ne.0) iarg=2
           if (index(cline(ix1:ix2),'DELGAM').ne.0) iarg=3
           if (index(cline(ix1:ix2),'EMITX').ne.0) iarg=4
           if (index(cline(ix1:ix2),'EMITY').ne.0) iarg=5
           if (index(cline(ix1:ix2),'XBEAM').ne.0) iarg=8   ! XBEAM can be also found in RXBEAM and PXBEAM
           if (index(cline(ix1:ix2),'YBEAM').ne.0) iarg=9   ! Thus comparison with XBEAM has to come first
           if (index(cline(ix1:ix2),'RXBEAM').ne.0) iarg=6
           if (index(cline(ix1:ix2),'RYBEAM').ne.0) iarg=7
           if (index(cline(ix1:ix2),'BETAX').ne.0) iarg=-6
           if (index(cline(ix1:ix2),'BETAY').ne.0) iarg=-7
           if (index(cline(ix1:ix2),'PXBEAM').ne.0) iarg=10
           if (index(cline(ix1:ix2),'PYBEAM').ne.0) iarg=11
           if (index(cline(ix1:ix2),'XPRIME').ne.0) iarg=-10
           if (index(cline(ix1:ix2),'YPRIME').ne.0) iarg=-11
           if (index(cline(ix1:ix2),'ALPHAX').ne.0) iarg=12
           if (index(cline(ix1:ix2),'ALPHAY').ne.0) iarg=13
           if (index(cline(ix1:ix2),'CURPEAK').ne.0) iarg=14
           if (index(cline(ix1:ix2),'ELOSS').ne.0) iarg=15
c
           if (iarg.eq.0) then
              do j=1,15
                 ipar(j)=j
              enddo   
              ncol=15
              j=printerr(errinwarn,
     c              'Unrecognized information line in BEAMFILE')
              return
           else
              ncol=ncol+1 
              ipar(ncol)=iarg
              if (abs(iarg).eq.1) haszpos=1
              cline=cline(ix2+1:255)
              if (ncol.lt.15) goto 1
           endif
         endif
         if (haszpos.lt.1) then
            do j=1,15
               ipar(j)=j
            enddo   
            ncol=15
            j=printerr(errinwarn,
     c            'ZPOS/TPOS column not specified in BEAMFILE')
            return
         endif
         return
      endif
      i=printerr(errinwarn,'Unrecognized information line in BEAMFILE')
      return
      end
c
      function readfield(cin,irec)
c     ==============================================
c     read field from input file
c     ----------------------------------------------
c
      include 'genesis.def'
      include 'io.cmn'
      include 'input.cmn'
      include 'field.cmn'
      include 'sim.cmn'
c
      complex*16 cin(ncar*ncar)
      integer ix,irec
      real*8 scltmp
c
      scltmp=xks*dsqrt(vacimp)/(dxy*eev*xkper0)
c
      readfield=noerr
      read(nfin,rec=irec,err=1) (cin(ix),ix=1,ncar*ncar)
      do ix=1,ncar*ncar
          cin(ix)=cin(ix)*scltmp
      enddo
c  
      return
 1    readfield=printerr(errreadfld,fieldfile)
      return
      end
c
c
      subroutine importdispersion 
c     ================================================================= 
c     apply dispersion to imported beam file from readpart
c     subroutine supplied by Atoosa Meseck from Bessy
c     ------------------------------------------------------------------
c
      include 'genesis.def'
      include 'input.cmn'
      include 'particle.cmn'
c
      real*8 iarho,iaphi,mam,imagl_old
      real*8 ypart_old,py_old
      real*8 ma12,ma33,ma34,ma43,ma56
      integer i,ierr
c
      if (ibfield.eq.0.d0) return                     
      if (idril.lt.0.d0) then                      
          ierr=printerr(errinwarn,'IDRIL<0-NO DISP.SECTION')                                     
          return
      endif  
c
      if (igamref.le.0) igamref=gamma0  
c          
      ibfield=abs(ibfield)      
c      
      imagl_old=imagl
      iarho=(igamref*5.11e-04)/(ibfield*0.299793)
      iaphi= asin(imagl/iarho)
      mam=tan(iaphi)/iarho
      imagl=iaphi*iarho 

      ma12=3*idril+4*iarho*sin(iaphi)*cos(iaphi)
     +      +2*idril*cos(iaphi)*cos(iaphi)
           
      ma33= mam*(-10*idril-8*imagl)+
     +      mam*mam*(26*imagl*idril+
     +      15*idril*idril+8*imagl*imagl)+
     +      mam**3*(-12*idril*(imagl**2)-
     +      20*(idril**2)*imagl-7*(idril)**3)
     +      +1+(mam**4)*(4*(idril)**3*imagl+
     +      (2*idril*imagl)**2+idril**4)
           
c ma44=ma33 
           
      ma34=mam*(-28*idril*imagl-
     +      8*(imagl)**2-20*(idril)**2)+
     +      mam*mam*(44*imagl*(idril)**2+
     +      21*(idril)**3+20*idril*imagl**2)+
     +      mam**3*(-16*(idril*imagl)**2-
     +      24*(idril**3)*imagl-8*(idril)**4)
     +      +(mam**4)*(4*(idril)**3*imagl**2+
     +      4*idril**4*imagl+idril**5)+
     +      5*idril+4*imagl

      ma43=tan(iaphi)*(-2*iarho+
     +      2*tan(iaphi)*imagl+tan(iaphi)*idril)*
     +      (2*iarho**2-4*tan(iaphi)*imagl*iarho-
     +      4*tan(iaphi)*idril*iarho+
     +      2*idril*tan(iaphi)*tan(iaphi)*imagl+
     +      (tan(iaphi)*idril)**2)/
     +      iarho**4 


      ma56=8*iarho*sin(iaphi)-
     +      4*iarho*sin(iaphi)*cos(iaphi)+2*idril-
     +      2*idril*cos(iaphi)*cos(iaphi)-4*imagl+
     +      ((5*idril+4*imagl)/igamref**2) 
c
      imagl=imagl_old
c
      do i=1,npart 
           theta(i)=theta(i)+
     +      (ma56*(gamma(i)-igamref)/igamref)*twopi/xlamds/convharm
           xpart(i)=xpart(i)+ma12*px(i)/gamma(i)
           ypart_old=ypart(i)
           py_old= py(i)
           ypart(i)=ma33*ypart_old+ma34*py_old/gamma(i)         
           py(i)=ma43*ypart_old*gamma(i)+ma33*py_old 
      enddo                                       
      return
      end !of import dispersion
c
c      
      subroutine importtransfer 
c     ================================================================= 
c     Transfer matrix calculation supplied by A. Meseck. 
c     ------------------------------------------------------------------
c
      include 'genesis.def'
      include 'io.cmn'
      include 'input.cmn'
      include 'field.cmn'
      include 'sim.cmn'
      include 'particle.cmn'
c
      real*8 ypart_old,py_old,xpart_old,px_old
      real*8 gamma_old,theta_old    
      integer i,ierr
c
      if (trama.eq.0.d0) return
      if (igamref.le.0) igamref=gamma0 

      do i=1,npart 
c  Denormalize      
         px(i)=px(i)/gamma(i)
         py(i)=py(i)/gamma(i)
cc
         xpart_old=xpart(i)
         ypart_old=ypart(i)
         px_old= px(i)
         py_old= py(i)
         theta_old=theta(i)
         gamma_old= gamma(i)

            xpart(i)=itram11*xpart_old+itram12*px_old+
     +      itram13*ypart_old+itram14*py_old+
     +      itram15*theta(i)*xlamds*convharm/twopi+
     +      itram16*(gamma(i)-igamref)/igamref

           px(i)=itram21*xpart_old+itram22*px_old+
     +      itram23*ypart_old+itram24*py_old+
     +      itram25*theta_old*xlamds*convharm/twopi+
     +      itram26*(gamma(i)-igamref)/igamref

         
           ypart(i)=itram31*xpart_old+itram32*px_old+
     +      itram33*ypart_old+itram34*py_old+
     +      itram35*theta_old*xlamds*convharm/twopi+
     +      itram36*(gamma(i)-igamref)/igamref

           py(i)=itram41*xpart_old+itram42*px_old+
     +      itram43*ypart_old+itram44*py_old+
     +      itram45*theta_old*xlamds*convharm/twopi+
     +      itram46*(gamma(i)-igamref)/igamref

          theta(i)=itram55*theta_old+ (itram56*
     +    ((gamma(i)-igamref)/igamref)*twopi/xlamds/convharm)+
     +    (itram51*xpart_old+itram52*px_old+itram53*ypart_old+
     +     itram54*py_old)*twopi/xlamds/convharm

         gamma(i)=(itram61*xpart_old+itram62*px_old+
     +      itram63*ypart_old+itram64*py_old+
     +      itram65*theta_old*xlamds*convharm/twopi)*
     +      igamref + itram66*(gamma(i)-igamref)+igamref

c normalization
           px(i)=px(i)*gamma(i)
           py(i)=py(i)*gamma(i)
cc
      enddo    
                                                                 
      return
      end !of import transfermatrix
c
c            
      function readpart(islice)
c     =================================================================
c     load complete set of particle from file
c     -----------------------------------------------------------------
c
      include 'genesis.def'
      include 'input.cmn'
      include 'particle.cmn'
      include 'io.cmn' 
      include 'sim.cmn'
c
      integer j,i,islice,idel
c

      logical isop
c
      readpart=noerr
c
      if (multconv.eq.0) then
        j=6*(islice-1)+1              ! reading every slice
      else  
        j=6*((islice-1) / convharm)+1 ! reading every (1/convharm)th slice 
      endif
c      
      read(npin,rec=j,  err=100) (gamma(i),i=1,npart0)
      read(npin,rec=j+1,err=100) (theta(i),i=1,npart0)
      read(npin,rec=j+2,err=100) (xpart(i),i=1,npart0)
      read(npin,rec=j+3,err=100) (ypart(i),i=1,npart0)
      read(npin,rec=j+4,err=100) (px(i),i=1,npart0)
      read(npin,rec=j+5,err=100) (py(i),i=1,npart0)
c
c     apply transfermatrix to particle distribution
      call importtransfer 
c  
c     dispersive section
      call importdispersion    
c
c
c     convert to higher harmonic
c
      if (convharm.gt.1) then
            do i=1,npart
             theta(i)=float(convharm)*theta(i)
            enddo 
      endif
c
c     calculate init. perpendicular velocity (needed in first call of track)
c
      do i=1,npart0      
        xpart(i)=xpart(i)*xkper0
        ypart(i)=ypart(i)*xkper0
	    btpar(i)=dsqrt(1.d0-(px(i)**2+py(i)**2+1.)/gamma(i)**2)     !parallel velocity
      enddo
c
c     check for particle losses from previous run
c
      idel=0          
      do i=1,npart 
         if (gamma(i).gt.0.) then
            gamma(i-idel)=gamma(i)  
            theta(i-idel)=theta(i)  
            xpart(i-idel)=xpart(i)
            ypart(i-idel)=ypart(i)
            px(i-idel)=px(i)
            py(i-idel)=py(i)
         else    
            idel=idel+1
         endif
      enddo
      npart=npart0-idel
      xcuren=xcuren*float(npart)/float(npart0)
c
      return
 100  readpart=printerr(errread,partfile)
      call last
      return
      end
c

      subroutine readslice(nget,x,px,y,py,g,tmin,tmax)
c     =================================================================
c     read slice [tmin,tmax] of particle from distribution file
c     -----------------------------------------------------------------
c
      include 'genesis.def'
      include 'time.cmn'
      include 'io.cmn'

      character*255 line,cerr
      integer nget,n0,i,ix,ip
      real*8 x(*),px(*),y(*),py(*),g(*)
      real*8 tx,tpx,ty,tpy,tt,tg
      real*8 tmin,tmax,values(10)
c
      nget=0
c
      if (keepdist.ne.0) then
         do ip=1,ndist        
           if ((distt(ip).ge.tmin).and.(distt(ip).le.tmax)) then !in slice
             nget=nget+1
             x(nget) =distx(ip)                        !add to raw distribution
             px(nget)=distpx(ip)
             y(nget) =disty(ip)
             py(nget)=distpy(ip)
             g(nget) =distgam(ip)
           endif   
         enddo
         return
      endif
c
      n0=-1
c
 1    read(ndis,200,err=100,end=50) line
c
c     processing line 
c      
      call touppercase(line)
      call getfirstchar(line,ix)
      if (ix.eq.0) goto 1             !empty line
      if (line(ix:ix).eq.'#') goto 1  !comment line
      if (line(ix:ix).eq.'?') goto 1  !information allready processed
c
      if ((n0.lt.0).and.(distversion.lt.1.0)) then
         i=extractval(line,values,1)
         if (i.lt.0) then
            write(cerr,*) 'DISTFILE has invalid input line: ',line
            i=printerr(errinput,cerr)
            call last
         endif
         n0=nint(values(1))
         goto 1
      endif
c
c     get record-tuple 
c
      i=extractval(line,values,ncoldis)
      if (i.lt.0) then
          write(cerr,*) 'DISTFILE has invalid input line: ',line
          i=printerr(errinput,cerr)
          call last
      endif
      do i=1,ncoldis
         if (icolpar(i).eq.1) tx =values(i)
         if (icolpar(i).eq.2) tpx=values(i)
         if (icolpar(i).eq.3) ty =values(i)
         if (icolpar(i).eq.4) tpy=values(i)
         if (icolpar(i).eq.5) tt =distrev*values(i)
         if (icolpar(i).eq.6) tg =values(i)
      enddo
      if (iconv2t.ne.0 ) tt=-tt/3.e8           !convert from z to t
      if (iconv2g.ne.0 ) tg=dsqrt(tg*tg+1.d0)  !convert from p to gamma
      if (iconv2px.ne.0) tpx=tpx*tg            !convert from x' to px
      if (iconv2py.ne.0) tpy=tpy*tg            !convert from y' to py
c
      if ((tt.ge.tmin).and.(tt.le.tmax)) then !in slice
           nget=nget+1
           x(nget) =tx                        !add to raw distribution
           px(nget)=tpx
           y(nget) =ty
           py(nget)=tpy
           g(nget) =tg
      endif   
      goto 1
c      
 50   rewind(ndis)              !set file pointer back
      return 
c
 100  i=printerr(errread,'DISTFILE')
      call last
      return
c
 200  format(a)
c
      end
c
      function readdistfile(file,nio)
c     ==================================================================
c     open an external file containing the distribution. read the file 
c     geting the parameter range etc.
c     ------------------------------------------------------------------
c
      include 'genesis.def'
      include 'particle.cmn'
      include 'input.cmn'
      include 'time.cmn'
      include 'io.cmn'
c
      character*(*) file
      character*255 cerr,line
      integer   nio,nget,i,ix,ip,niotmp
      real*8 tt,values(10)
c      
      ndistsize=-1
      nget=0
c
c     default settings
c
      iconv2g=1       !needs to convert from p to gamma
      iconv2t=0       !long coordinate is time
      iconv2px=0      !convert from x' to px
      iconv2py=0      !convert from y' to py
      distversion=0.1 !first number is number of particles
      distrev=1.      !normal order of long. position
      ncoldis=6       !6 dimension
      do i=1,10
         icolpar(i)=i ! x,px,y,py,t,gamma,id order
      enddo
c      
      readdistfile=-1
      if (index(file,' ').eq.1) return   !no file selected
c
      ftdist=detectfiletype(file)      ! check for filetype
c
csven some compiler gives an error here (segmentation fault)
c     
      niotmp=opentextfile(file,'old',nio)
      readdistfile=niotmp
      if (niotmp.lt.0) call last
c
 1    read(nio,200,err=100,end=50) line
c
c     processing line 
c      
      call touppercase(line)
      call getfirstchar(line,ix)
      if (ix.eq.0) goto 1             !empty line
      if (line(ix:ix).eq.'#') goto 1  !comment line
      if (line(ix:ix).eq.'?') then    !read information line
         call getdistfileinfo(line)
         goto 1
      endif
c
      if ((ndistsize.lt.0).and.(distversion.lt.1.0)) then
         i=extractval(line,values,1)
         if (i.lt.0) then
            cerr='DISTFILE has invalid input line:'//line
            i=printerr(errinput,cerr)
            call last
         endif
         ndistsize=nint(values(1))
         goto 1
      endif
c
c     get record-tuple 
c
      i=extractval(line,values,ncoldis)
      if (i.lt.0) then
          cerr='distfile has invalid input line:'//line
          i=printerr(errinput,cerr)
          call last
      endif
c
      nget=nget+1
      if (keepdist.ne.0) then    ! save variabls if kept in memory
        if (nget.gt.ndmax) then  ! check for overflow 
          i=printerr(errinput,'DISTFILE size exceeds NDMAX')
          call last
        endif 
        do i=1,ncoldis
           if (icolpar(i).eq.1) distx(nget)  =values(i)
           if (icolpar(i).eq.2) distpx(nget) =values(i)
           if (icolpar(i).eq.3) disty(nget)  =values(i)
           if (icolpar(i).eq.4) distpy(nget) =values(i)
           if (icolpar(i).eq.5) distt(nget)  =distrev*values(i)
           if (icolpar(i).eq.6) distgam(nget)=values(i)
        enddo
      endif

      do i=1,ncoldis
         if (icolpar(i).eq.5) tt=distrev*values(i)    !catch time value
      enddo
      if (iconv2t.ne.0) tt=-tt/3.e8           !convert from z to t
c
      if (nget.eq.1) then
         tdmin=tt
         tdmax=tt
      endif
      if (tt.lt.tdmin) tdmin=tt              !adjust min and max
      if (tt.gt.tdmax) tdmax=tt
      goto 1
c
 50   rewind(nio)               !go back to file beginning
      if (keepdist.ne.0) then
         close(nio)
         ndist=nget
         if (iconv2t.ne.0) then
            do ip=1,nget
              distt(ip)=-distt(ip)/3.e8   !convert from space coordinate
            enddo
         endif
         if (iconv2g.ne.0) then
            do ip=1,nget
              distgam(ip)=dsqrt(distgam(ip)*distgam(ip)+1.d0)  !convert from p to gamma
            enddo
         endif
         if (iconv2px.ne.0) then
            do ip=1,nget
              distpx(ip)=distpx(ip)*distgam(ip) ! convert from x' to px
            enddo
         endif
         if (iconv2py.ne.0) then
            do ip=1,nget
              distpy(ip)=distpy(ip)*distgam(ip) ! convert from y' to py
            enddo
         endif
      endif
c
c     set time window
c
      if (nslice.le.0) then
         nslice=int((tdmax-tdmin)*3d8/xlamds/zsep)
         ntail=0
      endif
c  
      if ((ndistsize.ge.0).and.(nget.ne.ndistsize)) then
         i=printerr(errinwarn,'DISTFILE has fewer lines than defined')
      endif
c
      if (ndcut.le.0) ndcut=nget*nharm/npart    !self optimizing
      if (ndcut.le.0) ndcut=1
c
      if (charge.le.0) then
         i=printerr(errinput,'CHARGE for DISTFILE is not defined')
         call last
      endif
c
      delcharge=charge/dble(nget)
      dtd=(tdmax-tdmin)/dble(ndcut)
c
      return
c
 100  readdistfile=printerr(errread,file)
      call last
      return
c
 200  format(a)
c
      end
c

      subroutine getdistfileinfo(line)
c     =================================================================
c     extract information from distfile
c     -----------------------------------------------------------------
c
      include 'genesis.def'
      include 'input.cmn'
      include 'particle.cmn'
      include 'io.cmn'
c
      character*(*) line
      character*255 cline
      integer n,i,idx,ix1,ix2,ncount,narg,iarg,j,ierr
      real*8  val
c
      narg=0          ! some compile initialize it to unity.
c
      n=len(line)
c      call getfirstchar(line,idx) ! get first character should be identical to index !!
c
c     version number
c  
      i=index(line,'VERSION') !check for version number
      if (i.gt.0) then
        ierr=extractnumber(line(i+7:n),val)
        if (ierr.lt.0) then
           i=printerr(errinwarn,
     c               'Unrecognized information line in distfile')
        else
           distversion=val
        endif
        return
      endif 
c
c     order of distributon
c
      i=index(line,'REVERSE')
      if (i.gt.0) then
         distrev=-1.
         return 
      endif   
c
c     beam charge
c
      i=index(line,'CHARGE')
      if (i.gt.0) then
        ierr=extractnumber(line(i+6:n),val)
        if (ierr.lt.0) then
           i=printerr(errinwarn,
     c               'Unrecognized information line in distfile')
        else
           charge=val
        endif
        return
      endif 
c
c     cuts in long. phase space
c
c     disabled - no program specific instruction should be present
c
c
c      i=index(line,'ndcut') 
c      if (i.gt.0) then
c        ierr=extractnumber(line(i+5:n),val)
c        if (ierr.lt.0) then
c           i=printerr(errinwarn,
c     c               'unrecognized information line in distfile')
c        else
c           ndcut=nint(val)
c        endif
c        return
c      endif 
c
c     size of record
c
      i=index(line,'SIZE') 
      if (i.gt.0) then
        ierr=extractnumber(line(i+4:n),val)
        if (ierr.lt.0) then
           i=printerr(errinwarn,
     c               'Unrecognized information line in distfile')
        else
           ndistsize=nint(val)
        endif
        return
      endif 
c
c     get order and dimension
c

      i=index(line,'COLUMNS') !check for colums headers
      if (i.gt.0) then
         do j=1,10
            icolpar(j)=0
         enddo
         ncoldis=0
         ncount=0
         cline=line(i+7:len(line))
 1       call getfirstchar(cline,ix1)
         if (ix1.gt.0) then
           ix2=255             !search backwards
           do j=255,ix1+1,-1   !for first space after ix1
             if (cline(j:j).eq.' ') ix2=j  
           enddo
           iarg=0
           if (index(cline(ix1:ix2),'P').ne.0) iarg=6
           if (index(cline(ix1:ix2),'GAMMA').ne.0) iarg=-6
           if (index(cline(ix1:ix2),'X').ne.0) iarg=1
           if (index(cline(ix1:ix2),'PX').ne.0) iarg=2
           if (index(cline(ix1:ix2),'XPRIME').ne.0) iarg=-2
           if (index(cline(ix1:ix2),'Y').ne.0) iarg=3
           if (index(cline(ix1:ix2),'PY').ne.0) iarg=4
           if (index(cline(ix1:ix2),'YPRIME').ne.0) iarg=-4
           if (index(cline(ix1:ix2),'T').ne.0) iarg=5
           if (index(cline(ix1:ix2),'Z').ne.0) iarg=-5
c
           ncount=ncount+1
           narg=narg+abs(iarg)
           ncoldis=ncoldis+1
           icolpar(ncoldis)=abs(iarg)
           if (iarg.eq.2)  iconv2px=0
           if (iarg.eq.-2) iconv2px=1
           if (iarg.eq.4)  iconv2py=0
           if (iarg.eq.-4) iconv2py=1
           if (iarg.eq.5)  iconv2t=0
           if (iarg.eq.-5) iconv2t=1
           if (iarg.eq.6)  iconv2g=1
           if (iarg.eq.-6) iconv2g=0
           cline=cline(ix2+1:255)
           if (ncount.lt.10) goto 1
         endif

         if (narg.ne.21) then
            j=printerr(errinwarn,
     c            'Not all dimensions defined in DISTFILE')
            call last
         endif
         return
      endif
c
c     unrecognized
c
      i=printerr(errinwarn,'Unrecognized information line in distfile')
      return
      end
c
      subroutine preset
c     ==================================================================
c     sets default values of program inputs.
c     default parameter modelled after the pegasus fel      
c     ------------------------------------------------------------------
c
      include 'genesis.def'
      include 'input.cmn'
      include 'field.cmn'
c
      integer i
c
c     wiggler parameters:
c
      aw0    = 0.735          !dimensionless wiggler amplitude (rms).
      xkx       = 0.0d0       !weak focusing strength in x-plane.
      xky       = 1.0d0       !weak focusing strength in x-plane.
      wcoefz(1) = 0.0d0       !field taper start location in z.
      wcoefz(2) = 0.0d0       !field taper gradient
      wcoefz(3) = 0.0d0       !field taper model
      delaw  = 0.0d0          !relative spread in aw
      iertyp = 0              !field error distribution type (<0 = correlated)
      iwityp = 0              !wiggler type (0=planar, 1= wiggler)
      awd    = 0.735          !virtual wiggler field for drift space
      iseed  = -1             !initial seed for wiggler error generation
      fbess0 = 0.0d0          !beam-radiation coupling
      xlamd  = 2.05d-2        !wiggler period  (m)
      awx    = 0.			  ! max offset in x for undulator misalignment
      awy    = 0.			  ! max offset in y for undulator misalignment
c
c     electron beam parameters:
c
      npart  = 8192           !number of simulation particles.
      gamma0 = 35.2d0         !electron beam lorentz factor (energy)
      delgam = 5.d-3          !rms energy (gamma) spread 
      rxbeam = 112.1d-6       !rms beam size in x-plane 
      rybeam = 112.1d-6       !rms beam size in y-plane 
      alphax= 0d0             !twiss alpha parameter in x
      alphay= 0d0             !twiss alpha parameter in y
      emitx  = 2.0d-6         !normalized emittance x-plane (pi m-rad)
      emity  = 2.0d-6         !normalized emittance y-plane (pi m-rad)
      xbeam  = 0.d0           !center in x (m)
      ybeam  = 0.d0           !center in y (m)
      pxbeam  = 0.d0          !center in px 
      pybeam  = 0.d0          !center in py
      cuttail = -1.           !no collimation transverse tail
      curpeak=2.5d2           !peak current
      conditx = 0.d0          !conditioning in x plane (1/m)
      condity = 0.d0          !conditioning in y plane (1/m)
      bunch = 0.d0            !prebunching, fraction
      bunchphase = 0.d0       !phase for prebunching
      emod = 0.d0             !energy modulation (in gamma)
      emodphase = 0.d0        !phase for energy modulation      
c
c     radiation:
c
      xlamds = 12.852d-6      !output radiation wavelength
      prad0  = 1.00d01        !input power
      pradh0 = 0.00d0         !input power of higher harmonics.
      zrayl  = .500d0         !rayleigh range (of radiation)
      zwaist = 0.0d0          !z location of (radiation) waist.
      radphase=0			  ! although not an input parameter, here is the best place to initialize it.
c
c     numerical control parameters:
c
      ildgam = 5              !energy loading parameter.
      ildpsi = 7              !phase loading parameter.
      ildx   = 1              !x-plane loading parameter.
      ildy   = 2              !y-plane loading parameter.
      ildpx  = 3              !x momentum loading parameter.
      ildpy  = 4              !y momentum loading parameter.
      itgaus = 1              !gaussian (<>0) or uniform (=0) loading
      nbins  = 4              !# of bins in the phase coordinate
      igamgaus = 1            !gaussian (<>0) or uniform (=0) enegy loading    
      inverfc  = 0            !<>0 uses inverse error function to load Gauss  
c
c     mesh discretization:
c
      ncar   = 151            !mesh points in one dimension (xy-grid).
      lbc    = 0              !boundary condition (0=diriqlet,<>0 neumann)
      rmax0  = 9.d0           !maximum edge of grid.
      nscr   = 0              !# radial modes for space charge
      nscz   = 0              !# longitudinal modes for space charge 
      nptr   = 40             !radial grid points for space charge
      dgrid  = 0.d0           !grid size(-dgrid to dgrid) if dgrid > 0
      rmax0sc= 0              !if <0 the grid for space charge calculation is fixed, defines the total gridsize  
      iscrkup= 0              !if <> 0 then space charge is calculated at each Runge-Lutta integratio step.
c
c     integration control parameter:
c
      nwig =  98              !number of undulator periods per module
      zsep =  1.              !seperation of slices in units of xlamds
      delz =  1.0             !integration step in units of xlamd
      nsec = 1                !number of sections    
      iorb = 0                !orbit correction flag (<>0 -> use term).
      zstop=-1.               !stop simulation @ z=zstop 
      magin=0                 !read magnetic field description if magin <>0
      magout=0                !write magnetic field description if magout <>0, stop for < 0
      version=0.1             !assume oldest version
c
c     scan control parameter
c
      iscan=0                 !>0 -> iscanth parameter selected
      nscan=3                 !number of scans  
      svar=0.01               !rel. variation [-sval,sval]                
      scan=' '                !scan defined by name
c
c     output control parameters:
c
      iphsty = 1              !history - steps in z 
      ipradi = 0              !radiation field - steps in z
      ippart = 0              !particles - steps in z
      ishsty = 1              !history - steps in t
      isradi = 0              !radiation field - steps in t
      ispart = 0              !particles - steps in t
      do i=1,11
         lout(i)=1            !flags for output
      enddo
      lout(6)=0               !disable diffraction calculation
      do i=12,14+1*nhmax
         lout(i)=0 
      enddo
      iotail=0                !<>0 => output include also slippage
      idump=0                 !<>0 => dump complete radiation field.
      nharm=1                 !# or harmonics in the bunching factor
      iallharm=0              !<>0 => all higher harmonics are calculated up to nharm 
      iharmsc=0               !<>0 => include effects on electron motion
      idmpfld=0               !<>0 => dump radiation field
      idmppar=0               !<>0 => dump particle distribution, with >1 on harmonics
      ilog=0                  !<>0 => terminal output written to file
      ffspec=0                !>0 => total power in near field
      						  !<0 => on-axis intensity in far field
      						  !=0 => on-axis intensity in near field
c
c     file names
c   
      beamfile=' '            !beam description fiel
      fieldfile=' '           !input radiation field
      maginfile=' '           !input magnetic field
      magoutfile=' '          !output magnetic field
      outputfile='template.out'          !output file
      partfile=' '            !input particle file
      distfile=' '            !input distribution file
      radfile=' '             !input radiation file
      filetype='ORIGINAL'     !filetype for output files (sdds,xml)
c
c     focussing:
c
      quadf = 1.23d0          !quad focus strength
      quadd = 0.00d0          !quad defocus strength
      qfdx  = 0.0             !quadrupole offset in x
      qfdy  = 0.0             !quadrupole offset in y
      fl    = 98.             !focus section length
      dl    = 0.              !defocus section length
      drl   = 0.              !drift length
      f1st  = 0.              !start fodo at this point 
      solen = 0.              !strength of solenoid field
      sl    = 0.              !solenoid length
c
c     time dependency parameters:
c
      curlen=1.d-3            !rms bunch length in m (<0 -> uniform beam)
      ntail=-253              !starting slice in beam relative to current peak
      nslice=408              !#slices
      itdp=0                  !<>0 => time dependent code
      shotnoise=1.d0          !scaling of the shotnoise meanvalue
      iall=0                  !<>0 => load phasespace with same loading
      ipseed=-1               !seeding of the random numbers for shotnoise
      ndcut=-1                !=<0 self optimized binning of ext. dist.
      isntyp=0                !=0 -> fawley algorithm <>0 -> Penman algorithm
c
c     extensions
c
      isravg=0                !<>0 enables energy loss by incorerent radiation
      isrsig=0                !<>0 enables growth of energy spread by inc. sr.
      eloss=0.0               !energy loss per meter
      convharm=1              !the harmonic to convert
      multconv=0              !<>0 imported + converted slice is used multiple times
      ibfield=0.0			  !field strength of magnetic chicane
      imagl=0.0				  !length of bending magnet of chicane
      idril=0.0               !length of drift between bending magnets
      igamref=0.0             ! length of reference energy of compressor
      alignradf=0			  !<>0 imported radfile is aligned to electron beam
      offsetradf=0            ! if aligned, number of slices to skip
c
c    transfermatrix 
c
      trama=0                 !<>0 allows to transform 6D phase space
      itram11=1.0d0           ! transfer matrix set equal to identity matrix  
      itram12=0.0d0
      itram13=0.0d0
      itram14=0.0d0
      itram15=0.0d0
      itram16=0.0d0
      itram21=0.0d0
      itram22=1.0d0
      itram23=0.0d0
      itram24=0.0d0
      itram25=0.0d0
      itram26=0.0d0
      itram31=0.0d0
      itram32=0.0d0
      itram33=1.0d0
      itram34=0.0d0
      itram35=0.0d0
      itram36=0.0d0
      itram41=0.0d0
      itram42=0.0d0
      itram43=0.0d0
      itram44=1.0d0
      itram45=0.0d0
      itram46=0.0d0
      itram51=0.0d0
      itram52=0.0d0
      itram53=0.0d0
      itram54=0.0d0
      itram55=1.0d0
      itram56=0.0d0
      itram61=0.0d0
      itram62=0.0d0
      itram63=0.0d0
      itram64=0.0d0
      itram65=0.0d0
      itram66=1.0d0
c 
      return
      end     !preset
c
c
c     
c
      subroutine readradfile(file)
c     =============================================================
c     read the file for external description of the radiation beam
c     -------------------------------------------------------------
c
      include 'genesis.def'
      include 'input.cmn'
      include 'time.cmn'
      include 'io.cmn'
c
      integer nin,i,ncol,ipar(5),ix,idata,j,ft,itmp
      character*(*) file
      character*50 cerr
      character*511 line
      real*8  ver,values(5),tmin,tmax,reverse,zoffset
c
      itmp=0
      nraddata=-1           
      if (index(file,' ').eq.1) return
c
      if ((iscan.gt.0).and.(iscan.lt.25)) then
          i=printerr(errscanex,file)
          return
      endif    
c
c     read file
c
      ft=detectfiletype(file)        ! check for filetype
c
      nin=opentextfile(file,'old',8)
      if (nin.lt.0) call last ! stop program on error
c
      nraddata=-1            !# of rows not defined
      idata=0             !# of rows read
      ncol =5             !# of elements per line
      reverse=1.0         !tail for ZPOS and head for TPOS comes first in file
      zoffset=0.          !offset of shifting the radiation profile
      ver=0.1
      do i=1,ncol
         ipar(i)=i        !basic order of input
      enddo
c
 1    read(nin,100,err=20,end=50) line
c
c     processing line
c     
      call touppercase(line)
      call getfirstchar(line,ix)
c
      if (ix.eq.0) goto 1                     !empty line
      if (line(ix:ix).eq.'#') goto 1          !no comment used
      if (line(ix:ix).eq.'?') then
        call getradfileinfo(line,ipar,ncol,nraddata,ver,reverse,zoffset)  !check for add. info
         goto 1
      endif
c
      if ((nraddata.lt.0).and.(ver.lt.1.0)) then !old version
         i=extractval(line,values,1)
         if (i.lt.0) then
            write(cerr,*) 'Line number of RADFILE cannot be determined'
            i=printerr(errinput,cerr)
         endif
         nraddata=nint(values(1))
         goto 1
      endif
c
      i=extractval(line,values,ncol)
      if (i.lt.0) then
        write(cerr,*) 'RADFILE data line ',idata+1,' has bad format'
        i=printerr(errinput,cerr)
        call last
      endif
c
      idata=idata+1
c
c     set default values      
c
      tradpos(idata)=xlamds*zsep*idata    !can we make this the default?
      tzrayl(idata)=zrayl
      tzwaist(idata)=zwaist
      tprad0(idata)=prad0
      tradphase(idata)=0.			! default no detuning or chirp.
c
c     write over with input data
c
      do j=1,ncol
         if (ipar(j).eq.1 ) tradpos(idata)    = reverse*values(j)
         if (ipar(j).eq.-1) tradpos(idata)    =-reverse*values(j)*3e8  ! time input
         if (ipar(j).eq.2 ) tprad0(idata)   = values(j)
         if (ipar(j).eq.3 ) tzrayl(idata)   = values(j)
         if (ipar(j).eq.4 ) tzwaist(idata)  = values(j)
         if (ipar(j).eq.5 ) tradphase(idata)  = values(j)
      enddo 
c
c     check for unphysical parameters
c
      if (tprad0(idata).lt.0) then
         itmp=printerr(errinput,'Radiation power negative in RADFILE')  !abort
      endif 
c
      if (tzrayl(idata).lt.0) then
         itmp=printerr(errinput,'ZRAYL negative in RADFILE')  !abort
      endif 
c
      if (idata.eq.1) then
         tmin=tradpos(idata)
         tmax=tradpos(idata)
      else
         if (tradpos(idata).gt.tmax) tmax=tradpos(idata)
         if (tradpos(idata).lt.tmin) tmin=tradpos(idata)
      endif
c
      goto 1
c
 50   close(nin)
c 
      if ((nraddata.ge.0).and.(idata.ne.nraddata)) then
         i=printerr(errinwarn,'RADFILE has fewer lines than defined')
      endif
      nraddata=idata
      if (idata.lt.2) then
       i=printerr(errinput,'RADFILE contains less than 2 valid lines')
       call last
      endif
      if (itmp.ne.0) then
       call last
      endif
c
      if (ver.ge.1.0) then
        do i=1,nraddata
          tradpos(i)=tradpos(i)-tmin   !set time window to zero
        enddo
      endif
c
      do i=1,nraddata
          tradpos(i)=tradpos(i)+zoffset   !set time window to zero
      enddo


c
      if (nslice.le.0) then
         nslice=int((tmax-tmin)/xlamds/zsep)
         if (ver.ge.1.0) then
           ntail=int(zoffset/xlamds/zsep)
         else
           ntail=int((tmin+zoffset)/xlamds/zsep)
         endif
         write(ilog,110) nslice,ntail
      endif
c
      return
c
 20   i=printerr(errread,file)
      close(nin)
      call last
      return
c
c     format statement
c
 100  format(a)
 110  format('Auto-adjustment of time window:',/,
     c       'nslice=',i6,/ 
     c       'ntail =',i6)
c
      end     !readradfile
c 
      subroutine getradfileinfo(line,ipar,ncol,ndata,ver,reverse,
     +                          zoffset)
c     =================================================================
c     extract information from beamfile
c     -----------------------------------------------------------------
c
      include 'genesis.def'
c
      character*(*) line
      character*511 cline
      integer ndata,ipar(*),i,n,ncol,ierr,j,ix1,ix2,haszpos,iarg
      real*8  ver,val,reverse,zoffset
c
c     version number
c
      i=index(line,'VERSION') !check for version number
      n=len(line)
      if (i.gt.0) then
        ierr=extractnumber(line(i+7:n),val)
        if (ierr.lt.0) then
           i=printerr(errinwarn,
     c               'Unrecognized information line in RADFILE')
        else
           ver=val
        endif
        return
      endif 
c
c     reverse order
c
      i=index(line,'REVERSE')
      n=len(line)
      if (i.gt.0) then
         reverse=-1.
         return
      endif
c         
c     line numbers
c
      i=index(line,'SIZE') !check for size argument (aka ndata)
      if (i.gt.0) then
        ierr=extractnumber(line(i+4:n),val)
        if (ierr.lt.0) then
           i=printerr(errinwarn,
     c               'Unrecognized information line in RADFILE')
        else
           ndata=nint(val)
        endif
        return
      endif
c         
c     longitudinal offset
c
      i=index(line,'OFFSET') !check for size argument (aka ndata)
      if (i.gt.0) then
        ierr=extractnumber(line(i+6:n),val)
        if (ierr.lt.0) then
           i=printerr(errinwarn,
     c               'Unrecognized information line in RADFILE')
        else
           zoffset=val
        endif
        return
      endif
c
c     column order
c
      i=index(line,'COLUMNS') !check for colums headers
      if (i.gt.0) then
         do j=1,5
            ipar(j)=0
         enddo
         ncol=0
         cline=line(i+7:n)
         haszpos=0
 1       call getfirstchar(cline,ix1)
         if (ix1.gt.0) then
           ix2=255             !search backwards
           do j=255,ix1+1,-1   !for first space after ix1
             if (cline(j:j).eq.' ') ix2=j  
           enddo
           iarg=0
           if (index(cline(ix1:ix2),'ZPOS').ne.0) iarg=1
           if (index(cline(ix1:ix2),'TPOS').ne.0) iarg=-1
           if (index(cline(ix1:ix2),'PRAD0').ne.0) iarg=2
           if (index(cline(ix1:ix2),'ZRAYL').ne.0) iarg=3
           if (index(cline(ix1:ix2),'ZWAIST').ne.0) iarg=4
           if (index(cline(ix1:ix2),'PHASE').ne.0) iarg=5
c
           if (iarg.eq.0) then
              do j=1,5
                 ipar(j)=j
              enddo   
              ncol=5
              j=printerr(errinwarn,
     c              'Unrecognized information line in RADFILE')
              return
           else
              ncol=ncol+1 
              ipar(ncol)=iarg
              if (abs(iarg).eq.1) haszpos=1
              cline=cline(ix2+1:255)
              if (ncol.lt.5) goto 1
           endif
         endif
         if (haszpos.lt.1) then
            do j=1,5
               ipar(j)=j
            enddo   
            ncol=5
            j=printerr(errinwarn,
     c            'ZPOS/TPOS column not specified in RADFILE')
            return
         endif
         return
      endif
      i=printerr(errinwarn,'Unrecognized information line in RADFILE')
      return
      end
c


